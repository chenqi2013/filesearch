use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use parking_lot::{Mutex, RwLock};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

pub const EMBEDDING_DIMENSION: usize = 512;
const MODEL_NAME: &str = "BAAI/bge-small-zh-v1.5";

enum ModelState {
    Uninitialized,
    Ready(Box<TextEmbedding>),
    Fallback,
}

pub struct EmbeddingEngine {
    cache_dir: PathBuf,
    state: Mutex<ModelState>,
    status: RwLock<String>,
}

impl EmbeddingEngine {
    pub fn new(cache_dir: PathBuf) -> Self {
        let force_offline = std::env::var("FILESEARCH_EMBEDDING_OFFLINE")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"));
        Self {
            cache_dir,
            state: Mutex::new(if force_offline {
                ModelState::Fallback
            } else {
                ModelState::Uninitialized
            }),
            status: RwLock::new(if force_offline {
                format!("{MODEL_NAME}（强制离线特征）")
            } else {
                format!("{MODEL_NAME}（等待加载）")
            }),
        }
    }

    pub fn status(&self) -> String {
        self.status.read().clone()
    }

    pub fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let mut state = self.state.lock();
        if matches!(*state, ModelState::Uninitialized) {
            *self.status.write() = format!("{MODEL_NAME}（正在加载）");
            match self.load_model() {
                Ok(model) => {
                    *self.status.write() = format!("{MODEL_NAME}（本地 ONNX）");
                    *state = ModelState::Ready(Box::new(model));
                }
                Err(error) => {
                    tracing::warn!(%error, "local embedding model unavailable; using offline fallback");
                    *self.status.write() = format!("{MODEL_NAME}（离线特征降级）");
                    *state = ModelState::Fallback;
                }
            }
        }

        if let ModelState::Ready(model) = &mut *state {
            match model.embed(texts, Some(32)) {
                Ok(vectors) => return vectors,
                Err(error) => {
                    tracing::warn!(%error, "embedding inference failed; using offline fallback");
                    *self.status.write() = format!("{MODEL_NAME}（推理失败，已降级）");
                    *state = ModelState::Fallback;
                }
            }
        }
        texts.iter().map(|text| fallback_embed(text)).collect()
    }

    fn load_model(&self) -> Result<TextEmbedding> {
        std::fs::create_dir_all(&self.cache_dir)
            .with_context(|| format!("无法创建模型目录 {}", self.cache_dir.display()))?;
        let options = TextInitOptions::new(EmbeddingModel::BGESmallZHV15)
            .with_cache_dir(self.cache_dir.clone())
            .with_max_length(512)
            .with_show_download_progress(false);
        TextEmbedding::try_new(options).context("无法加载中文 Embedding 模型")
    }
}

pub fn fallback_embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIMENSION];
    for token in tokens(text) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();
        let index = (hash as usize) % EMBEDDING_DIMENSION;
        vector[index] += if hash & 1 == 0 { 1.0 } else { -1.0 };
    }
    normalize(&mut vector);
    vector
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
