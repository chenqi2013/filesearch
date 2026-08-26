use anyhow::{Context, Result};
use ort::operator::{
    io::{OperatorInput, OperatorOutput},
    kernel::{Kernel, KernelAttributes, KernelContext},
    Operator, OperatorDomain,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::Tensor;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

pub const EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_PROFILE: &str = "rwkv-document-source-v2";
const MODEL_NAME: &str = "EmbeddingRWKV Tiny";
const EOS_TOKEN_ID: i64 = 65535;
const INFERENCE_BATCH_SIZE: usize = 4;
const RWKV_HEAD_COUNT: usize = 12;
const RWKV_HEAD_SIZE: usize = 64;

struct Rwkv7Operator;

impl Operator for Rwkv7Operator {
    fn name(&self) -> &str {
        "Rwkv7"
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        (0..6)
            .map(|_| OperatorInput::required(TensorElementType::Float32))
            .collect()
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, _: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        Ok(Box::new(|context: &KernelContext| {
            let inputs = (0..6)
                .map(|index| {
                    context
                        .input(index)?
                        .ok_or_else(|| ort::Error::new("EmbeddingRWKV WKV 输入缺失"))
                })
                .collect::<ort::Result<Vec<_>>>()?;
            let (shape, receptance) = inputs[0].try_extract_tensor::<f32>()?;
            if shape.len() != 3 || shape[2] != EMBEDDING_DIMENSION as i64 {
                return Err(ort::Error::new("EmbeddingRWKV WKV 输入维度无效"));
            }
            let batch_size = shape[0] as usize;
            let token_count = shape[1] as usize;
            let expected_values = batch_size * token_count * EMBEDDING_DIMENSION;
            let tensors = inputs[1..]
                .iter()
                .map(|input| input.try_extract_tensor::<f32>().map(|(_, values)| values))
                .collect::<ort::Result<Vec<_>>>()?;
            if tensors.iter().any(|values| values.len() != expected_values) {
                return Err(ort::Error::new("EmbeddingRWKV WKV 输入长度不一致"));
            }
            let decay = tensors[0];
            let key = tensors[1];
            let value = tensors[2];
            let in_context_key = tensors[3];
            let in_context_value = tensors[4];
            let mut output = context
                .output(0, shape.to_vec())?
                .ok_or_else(|| ort::Error::new("EmbeddingRWKV WKV 输出缺失"))?;
            let (_, output_values) = output.try_extract_tensor_mut::<f32>()?;
            rwkv7_forward(
                batch_size,
                token_count,
                receptance,
                decay,
                key,
                value,
                in_context_key,
                in_context_value,
                output_values,
            );
            Ok(())
        }))
    }
}

#[allow(clippy::too_many_arguments)]
fn rwkv7_forward(
    batch_size: usize,
    token_count: usize,
    receptance: &[f32],
    decay: &[f32],
    key: &[f32],
    value: &[f32],
    in_context_key: &[f32],
    in_context_value: &[f32],
    output: &mut [f32],
) {
    let state_size = RWKV_HEAD_COUNT * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
    let batch_stride = token_count * EMBEDDING_DIMENSION;
    if batch_size == 1 {
        rwkv7_forward_batch(
            token_count,
            receptance,
            decay,
            key,
            value,
            in_context_key,
            in_context_value,
            output,
            state_size,
        );
        return;
    }
    std::thread::scope(|scope| {
        let output_batches = output.chunks_exact_mut(batch_stride);
        for (batch_index, output_batch) in output_batches.enumerate() {
            let start = batch_index * batch_stride;
            let end = start + batch_stride;
            let receptance_batch = &receptance[start..end];
            let decay_batch = &decay[start..end];
            let key_batch = &key[start..end];
            let value_batch = &value[start..end];
            let in_context_key_batch = &in_context_key[start..end];
            let in_context_value_batch = &in_context_value[start..end];
            scope.spawn(move || {
                rwkv7_forward_batch(
                    token_count,
                    receptance_batch,
                    decay_batch,
                    key_batch,
                    value_batch,
                    in_context_key_batch,
                    in_context_value_batch,
                    output_batch,
                    state_size,
                );
            });
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn rwkv7_forward_batch(
    token_count: usize,
    receptance: &[f32],
    decay: &[f32],
    key: &[f32],
    value: &[f32],
    in_context_key: &[f32],
    in_context_value: &[f32],
    output: &mut [f32],
    state_size: usize,
) {
    let mut state = vec![0.0_f32; state_size];
    for token_index in 0..token_count {
        let token_offset = token_index * EMBEDDING_DIMENSION;
        for head_index in 0..RWKV_HEAD_COUNT {
            let vector_offset = token_offset + head_index * RWKV_HEAD_SIZE;
            let state_offset = head_index * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
            let mut projection = [0.0_f32; RWKV_HEAD_SIZE];
            for row in 0..RWKV_HEAD_SIZE {
                let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                let mut projected = 0.0_f32;
                for column in 0..RWKV_HEAD_SIZE {
                    let state_index = row_offset + column;
                    let decayed = state[state_index] * decay[vector_offset + column];
                    state[state_index] = decayed;
                    projected += decayed * in_context_key[vector_offset + column];
                }
                projection[row] = projected;
            }
            for row in 0..RWKV_HEAD_SIZE {
                let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                let mut mixed = 0.0_f32;
                for column in 0..RWKV_HEAD_SIZE {
                    let state_index = row_offset + column;
                    let updated = state[state_index]
                        + projection[row] * in_context_value[vector_offset + column]
                        + value[vector_offset + row] * key[vector_offset + column];
                    state[state_index] = updated;
                    mixed += updated * receptance[vector_offset + column];
                }
                output[vector_offset + row] = mixed;
            }
        }
    }
}

struct RwkvModel {
    session: Session,
    tokenizer: RwkvTokenizer,
}

#[derive(Default)]
struct TokenNode {
    children: HashMap<u8, usize>,
    token_id: Option<i64>,
}

struct RwkvTokenizer {
    nodes: Vec<TokenNode>,
}

enum ModelState {
    Uninitialized,
    Ready(Box<RwkvModel>),
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
        self.embed_batch(&[query.to_owned()])
            .into_iter()
            .next()
            .unwrap_or_else(|| fallback_embed(query))
    }

    pub fn embed_passages(&self, passages: &[String]) -> Vec<Vec<f32>> {
        self.embed_batch(passages)
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
                    *self.status.write() = format!("{MODEL_NAME}（内置 ONNX）");
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
            match model.embed(texts) {
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

    fn load_model(&self) -> Result<RwkvModel> {
        let model_path = self.model_dir.join("model.onnx");
        let vocab_path = self.model_dir.join("rwkv_vocab.bin");
        let tokenizer = RwkvTokenizer::load(&vocab_path)?;
        let operators = OperatorDomain::new("com.localfind")
            .context("无法创建 EmbeddingRWKV 自定义算子域")?
            .add(Rwkv7Operator)
            .context("无法注册 EmbeddingRWKV WKV 算子")?;
        let session = Session::builder()
            .context("无法初始化 ONNX Runtime")?
            .with_operators(operators)
            .context("无法配置 EmbeddingRWKV WKV 算子")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("无法配置 ONNX 图优化")?
            .commit_from_file(&model_path)
            .with_context(|| format!("无法加载 EmbeddingRWKV 模型 {}", model_path.display()))?;
        Ok(RwkvModel { session, tokenizer })
    }
}

impl RwkvModel {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(INFERENCE_BATCH_SIZE) {
            let mut token_batch = batch
                .iter()
                .map(|text| {
                    let mut tokens = self.tokenizer.encode(text).into_iter().collect::<Vec<_>>();
                    tokens.push(EOS_TOKEN_ID);
                    tokens
                })
                .collect::<Vec<_>>();
            let max_length = token_batch.iter().map(Vec::len).max().unwrap_or(1);
            let mut flattened = Vec::with_capacity(batch.len() * max_length);
            for tokens in &mut token_batch {
                flattened.extend(std::iter::repeat_n(0, max_length - tokens.len()));
                flattened.append(tokens);
            }
            let input = Tensor::<i64>::from_array(([batch.len(), max_length], flattened))
                .context("无法创建 EmbeddingRWKV 输入")?;
            let outputs = self
                .session
                .run(ort::inputs![input])
                .context("EmbeddingRWKV ONNX 推理失败")?;
            let (shape, values) = outputs[0]
                .try_extract_tensor::<f32>()
                .context("EmbeddingRWKV 输出格式无效")?;
            if shape.as_ref() != [batch.len() as i64, EMBEDDING_DIMENSION as i64] {
                anyhow::bail!("EmbeddingRWKV 输出维度异常: {shape:?}");
            }
            vectors.extend(values.chunks_exact(EMBEDDING_DIMENSION).map(|values| {
                let mut vector = values.to_vec();
                normalize(&mut vector);
                vector
            }));
        }
        Ok(vectors)
    }
}

impl RwkvTokenizer {
    fn load(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("无法读取 EmbeddingRWKV 词表 {}", path.display()))?;
        if bytes.get(..8) != Some(b"RWKVTOK1") {
            anyhow::bail!("EmbeddingRWKV 词表格式无效");
        }
        let mut offset = 8;
        let count = read_u32(&bytes, &mut offset)? as usize;
        let mut tokenizer = Self {
            nodes: vec![TokenNode::default()],
        };
        for token_id in 0..count {
            let length = read_u16(&bytes, &mut offset)? as usize;
            let end = offset
                .checked_add(length)
                .filter(|end| *end <= bytes.len())
                .context("EmbeddingRWKV 词表数据不完整")?;
            if token_id != 0 && length > 0 {
                tokenizer.insert(&bytes[offset..end], token_id as i64);
            }
            offset = end;
        }
        Ok(tokenizer)
    }

    fn insert(&mut self, token: &[u8], token_id: i64) {
        let mut node_index = 0;
        for byte in token {
            let next_index = if let Some(index) = self.nodes[node_index].children.get(byte) {
                *index
            } else {
                let index = self.nodes.len();
                self.nodes.push(TokenNode::default());
                self.nodes[node_index].children.insert(*byte, index);
                index
            };
            node_index = next_index;
        }
        self.nodes[node_index].token_id = Some(token_id);
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let mut node_index = 0;
            let mut cursor = offset;
            let mut longest = None;
            while let Some(next_index) = bytes
                .get(cursor)
                .and_then(|byte| self.nodes[node_index].children.get(byte))
            {
                node_index = *next_index;
                cursor += 1;
                if let Some(token_id) = self.nodes[node_index].token_id {
                    longest = Some((cursor, token_id));
                }
            }
            let (next_offset, token_id) = longest.expect("RWKV 词表必须覆盖每个 UTF-8 字节");
            tokens.push(token_id);
            offset = next_offset;
        }
        tokens
    }
}

fn read_u16(bytes: &[u8], offset: &mut usize) -> Result<u16> {
    let end = *offset + 2;
    let value = bytes
        .get(*offset..end)
        .context("EmbeddingRWKV 词表数据不完整")?;
    *offset = end;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32> {
    let end = *offset + 4;
    let value = bytes
        .get(*offset..end)
        .context("EmbeddingRWKV 词表数据不完整")?;
    *offset = end;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
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

    fn rwkv_tokenizer() -> RwkvTokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/embedding-rwkv-tiny/rwkv_vocab.bin");
        RwkvTokenizer::load(&path).unwrap()
    }

    #[test]
    fn rwkv_tokenizer_matches_official_world_tokenizer() {
        assert_eq!(
            rwkv_tokenizer().encode("本地文档搜索"),
            [13205, 11459, 13012, 13351, 12877, 15325]
        );
        assert_eq!(
            rwkv_tokenizer().encode("EmbeddingRWKV Tiny test"),
            [33071, 25139, 1413, 1184, 29906, 32223]
        );
    }

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
