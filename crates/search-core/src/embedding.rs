use anyhow::{Context, Result};
use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use parking_lot::{Mutex, RwLock};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

pub const EMBEDDING_DIMENSION: usize = 384;
const MODEL_NAME: &str = "intfloat/multilingual-e5-small";

enum ModelState {
    Uninitialized,
    Ready(Box<TextEmbedding>),
    Fallback,
}

pub struct EmbeddingEngine {
    model_dir: PathBuf,
    state: Mutex<ModelState>,
    status: RwLock<String>,
}

impl EmbeddingEngine {
    pub fn new(model_dir: PathBuf) -> Self {
        configure_onnx_runtime();
        let force_offline = std::env::var("FILESEARCH_EMBEDDING_OFFLINE")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"));
        Self {
            model_dir,
            state: Mutex::new(if force_offline {
                ModelState::Fallback
            } else {
                ModelState::Uninitialized
            }),
            status: RwLock::new(if force_offline {
                format!("{MODEL_NAME}（强制离线特征）")
            } else {
                format!("{MODEL_NAME}（内置模型待加载）")
            }),
        }
    }

    pub fn status(&self) -> String {
        self.status.read().clone()
    }

    pub fn embed_query(&self, query: &str) -> Vec<f32> {
        self.embed_batch(&[format!("query: {query}")])
            .into_iter()
            .next()
            .unwrap_or_else(|| fallback_embed(query))
    }

    pub fn embed_passages(&self, passages: &[String]) -> Vec<Vec<f32>> {
        let inputs = passages
            .iter()
            .map(|passage| format!("passage: {passage}"))
            .collect::<Vec<_>>();
        self.embed_batch(&inputs)
    }

    fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let mut state = self.state.lock();
        if matches!(*state, ModelState::Uninitialized) {
            *self.status.write() = format!("{MODEL_NAME}（正在加载内置模型）");
            match self.load_model() {
                Ok(model) => {
                    *self.status.write() = format!("{MODEL_NAME}（本地 ONNX）");
                    *state = ModelState::Ready(Box::new(model));
                }
                Err(error) => {
                    tracing::warn!(error = %format_args!("{error:#}"), "local embedding model unavailable; using offline fallback");
                    *self.status.write() = format!("{MODEL_NAME}（内置模型不可用，离线特征降级）");
                    *state = ModelState::Fallback;
                }
            }
        }

        if let ModelState::Ready(model) = &mut *state {
            match model.embed(texts, Some(32)) {
                Ok(vectors) => return vectors,
                Err(error) => {
                    tracing::warn!(error = %format_args!("{error:#}"), "embedding inference failed; using offline fallback");
                    *self.status.write() = format!("{MODEL_NAME}（推理失败，已降级）");
                    *state = ModelState::Fallback;
                }
            }
        }
        texts.iter().map(|text| fallback_embed(text)).collect()
    }

    fn load_model(&self) -> Result<TextEmbedding> {
        let read_model_file = |relative: &str| {
            std::fs::read(self.model_dir.join(relative)).with_context(|| {
                format!(
                    "无法读取内置模型文件 {}",
                    self.model_dir.join(relative).display()
                )
            })
        };
        let model = UserDefinedEmbeddingModel::new(
            read_model_file("onnx/model.onnx")?,
            TokenizerFiles {
                tokenizer_file: read_model_file("tokenizer.json")?,
                config_file: read_model_file("config.json")?,
                special_tokens_map_file: read_model_file("special_tokens_map.json")?,
                tokenizer_config_file: read_model_file("tokenizer_config.json")?,
            },
        )
        .with_pooling(Pooling::Mean);
        TextEmbedding::try_new_from_user_defined(
            model,
            InitOptionsUserDefined::new().with_max_length(512),
        )
        .context("无法加载内置多语言 Embedding 模型")
    }
}

pub fn fallback_embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIMENSION];
    for token in tokens(strip_e5_prefix(text)) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();
        let index = (hash as usize) % EMBEDDING_DIMENSION;
        vector[index] += if hash & 1 == 0 { 1.0 } else { -1.0 };
    }
    normalize(&mut vector);
    vector
}

fn configure_onnx_runtime() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let Some(executable_dir) = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(PathBuf::from))
    else {
        return;
    };
    let runtime_path = executable_dir.join("onnxruntime.dll");
    if runtime_path.is_file() {
        std::env::set_var("ORT_DYLIB_PATH", runtime_path);
    }
}

fn strip_e5_prefix(text: &str) -> &str {
    text.strip_prefix("query: ")
        .or_else(|| text.strip_prefix("passage: "))
        .unwrap_or(text)
}

pub fn lexical_terms(text: &str) -> Vec<String> {
    let mut values = tokens(text).into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    values
}

pub fn lexical_text(text: &str) -> String {
    lexical_terms(text).join(" ")
}

fn tokens(text: &str) -> HashSet<String> {
    let lower = text.to_lowercase();
    let mut output = HashSet::new();
    let mut word = String::new();
    let mut cjk = Vec::new();
    for character in lower.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            word.push(character);
            cjk.clear();
            continue;
        }
        if word.len() > 1 {
            output.insert(std::mem::take(&mut word));
        } else {
            word.clear();
        }
        if is_cjk(character) {
            cjk.push(character);
            output.insert(character.to_string());
            if cjk.len() >= 2 {
                output.insert(cjk[cjk.len() - 2..].iter().collect());
            }
        } else {
            cjk.clear();
        }
    }
    if word.len() > 1 {
        output.insert(word);
    }
    output
}

fn is_cjk(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x9fff | 0xf900..=0xfaff)
}

pub fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

pub fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return 0.0;
    }
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_embedding_still_handles_chinese_bigrams() {
        let query = fallback_embed("本地文档搜索");
        let related = fallback_embed("Windows 本地文档智能搜索工具");
        let unrelated = fallback_embed("季度餐饮费用报表");
        assert!(cosine(&query, &related) > cosine(&query, &unrelated));
    }

    #[test]
    fn lexical_terms_include_chinese_bigrams() {
        let terms = lexical_terms("本地搜索 Rust");
        assert!(terms.contains(&"本地".to_owned()));
        assert!(terms.contains(&"rust".to_owned()));
    }
}
