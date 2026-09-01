use anyhow::{Context, Result};
use ort::execution_providers::CUDAExecutionProvider;
use ort::operator::{
    io::{OperatorInput, OperatorOutput},
    kernel::{Kernel, KernelAttributes, KernelContext},
    Operator, OperatorDomain,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::tensor::TensorElementType;
use ort::value::Tensor;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

pub const EMBEDDING_DIMENSION: usize = 768;
pub const EMBEDDING_PROFILE: &str = "rwkv-document-source-v3";
const MODEL_NAME: &str = "EmbeddingRWKV Tiny";
const EOS_TOKEN_ID: i64 = 65535;
const INFERENCE_MAX_BATCH_SIZE: usize = 4;
const INFERENCE_MAX_PADDED_TOKENS: usize = 8_192;
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
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            rwkv7_forward_batch_avx2(
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
        }
        return;
    }
    rwkv7_forward_batch_scalar(
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
}

#[allow(clippy::too_many_arguments)]
fn rwkv7_forward_batch_scalar(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rwkv7_forward_batch_avx2(
    token_count: usize,
    receptance: &[f32],
    decay: &[f32],
    key: &[f32],
    value: &[f32],
    in_context_key: &[f32],
    in_context_value: &[f32],
    output: &mut [f32],
    _state_size: usize,
) {
    use std::arch::x86_64::*;

    let mut head_outputs = (0..RWKV_HEAD_COUNT)
        .map(|_| vec![0.0_f32; token_count * RWKV_HEAD_SIZE])
        .collect::<Vec<_>>();
    head_outputs
        .par_iter_mut()
        .enumerate()
        .for_each(|(head_index, head_output)| {
            let mut state = vec![0.0_f32; RWKV_HEAD_SIZE * RWKV_HEAD_SIZE];
            let state_offset = 0;
            for token_index in 0..token_count {
                let token_offset = token_index * EMBEDDING_DIMENSION;
                let vector_offset = token_offset + head_index * RWKV_HEAD_SIZE;
                let mut projection = [0.0_f32; RWKV_HEAD_SIZE];
                for row in 0..RWKV_HEAD_SIZE {
                    let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                    let mut projected = _mm256_setzero_ps();
                    for column in (0..RWKV_HEAD_SIZE).step_by(8) {
                        let state_ptr = state.as_mut_ptr().add(row_offset + column);
                        let decayed = _mm256_mul_ps(
                            _mm256_loadu_ps(state_ptr),
                            _mm256_loadu_ps(decay.as_ptr().add(vector_offset + column)),
                        );
                        _mm256_storeu_ps(state_ptr, decayed);
                        projected = _mm256_add_ps(
                            projected,
                            _mm256_mul_ps(
                                decayed,
                                _mm256_loadu_ps(
                                    in_context_key.as_ptr().add(vector_offset + column),
                                ),
                            ),
                        );
                    }
                    let halves = _mm_add_ps(
                        _mm256_castps256_ps128(projected),
                        _mm256_extractf128_ps(projected, 1),
                    );
                    let pairs = _mm_hadd_ps(halves, halves);
                    let singles = _mm_hadd_ps(pairs, pairs);
                    projection[row] = _mm_cvtss_f32(singles);
                }
                for row in 0..RWKV_HEAD_SIZE {
                    let row_offset = state_offset + row * RWKV_HEAD_SIZE;
                    let projection_value = _mm256_set1_ps(projection[row]);
                    let value_value = _mm256_set1_ps(value[vector_offset + row]);
                    let mut mixed = _mm256_setzero_ps();
                    for column in (0..RWKV_HEAD_SIZE).step_by(8) {
                        let state_ptr = state.as_mut_ptr().add(row_offset + column);
                        let updated = _mm256_add_ps(
                            _mm256_loadu_ps(state_ptr),
                            _mm256_add_ps(
                                _mm256_mul_ps(
                                    projection_value,
                                    _mm256_loadu_ps(
                                        in_context_value.as_ptr().add(vector_offset + column),
                                    ),
                                ),
                                _mm256_mul_ps(
                                    value_value,
                                    _mm256_loadu_ps(key.as_ptr().add(vector_offset + column)),
                                ),
                            ),
                        );
                        _mm256_storeu_ps(state_ptr, updated);
                        mixed = _mm256_add_ps(
                            mixed,
                            _mm256_mul_ps(
                                updated,
                                _mm256_loadu_ps(receptance.as_ptr().add(vector_offset + column)),
                            ),
                        );
                    }
                    let halves = _mm_add_ps(
                        _mm256_castps256_ps128(mixed),
                        _mm256_extractf128_ps(mixed, 1),
                    );
                    let pairs = _mm_hadd_ps(halves, halves);
                    let singles = _mm_hadd_ps(pairs, pairs);
                    head_output[token_index * RWKV_HEAD_SIZE + row] = _mm_cvtss_f32(singles);
                }
            }
        });
    for token_index in 0..token_count {
        let output_offset = token_index * EMBEDDING_DIMENSION;
        let head_offset = token_index * RWKV_HEAD_SIZE;
        for (head_index, head_output) in head_outputs.iter().enumerate() {
            let output_start = output_offset + head_index * RWKV_HEAD_SIZE;
            output[output_start..output_start + RWKV_HEAD_SIZE]
                .copy_from_slice(&head_output[head_offset..head_offset + RWKV_HEAD_SIZE]);
        }
    }
}

struct RwkvModel {
    session: Session,
    tokenizer: RwkvTokenizer,
    backend: RuntimeBackend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeBackend {
    Cpu,
    Cuda,
}

impl RuntimeBackend {
    fn code(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "NVIDIA CUDA（ONNX 节点，WKV CPU）",
        }
    }
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
    preferred_backend: RuntimeBackend,
    state: Mutex<ModelState>,
    status: RwLock<String>,
}

impl EmbeddingEngine {
    pub fn new(model_dir: PathBuf) -> Self {
        let preferred_backend = configure_onnx_runtime();
        let force_offline = std::env::var("FILESEARCH_EMBEDDING_OFFLINE")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE"));
        Self {
            model_dir,
            preferred_backend,
            state: Mutex::new(if force_offline {
                ModelState::Fallback
            } else {
                ModelState::Uninitialized
            }),
            status: RwLock::new(if force_offline {
                format!("{MODEL_NAME}（强制离线特征）")
            } else if preferred_backend == RuntimeBackend::Cuda {
                format!("{MODEL_NAME}（检测到 NVIDIA CUDA，待加载）")
            } else {
                format!("{MODEL_NAME}（CPU，内置模型待加载）")
            }),
        }
    }

    pub fn status(&self) -> String {
        self.status.read().clone()
    }

    pub fn backend(&self) -> String {
        let state = self.state.lock();
        match &*state {
            ModelState::Ready(model) => model.backend.code().to_owned(),
            ModelState::Fallback => "fallback".to_owned(),
            ModelState::Uninitialized => self.preferred_backend.code().to_owned(),
        }
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
                    *self.status.write() =
                        format!("{MODEL_NAME}（内置 ONNX，{}）", model.backend.label());
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
        if self.preferred_backend == RuntimeBackend::Cuda {
            match self.build_session(&model_path, RuntimeBackend::Cuda) {
                Ok(session) => {
                    return Ok(RwkvModel {
                        session,
                        tokenizer,
                        backend: RuntimeBackend::Cuda,
                    });
                }
                Err(error) => {
                    tracing::warn!(
                        error = %format_args!("{error:#}"),
                        "NVIDIA CUDA embedding backend unavailable; falling back to CPU"
                    );
                }
            }
        }
        let session = self.build_session(&model_path, RuntimeBackend::Cpu)?;
        Ok(RwkvModel {
            session,
            tokenizer,
            backend: RuntimeBackend::Cpu,
        })
    }

    fn build_session(
        &self,
        model_path: &std::path::Path,
        backend: RuntimeBackend,
    ) -> Result<Session> {
        let operators = OperatorDomain::new("com.localfind")
            .context("无法创建 EmbeddingRWKV 自定义算子域")?
            .add(Rwkv7Operator)
            .context("无法注册 EmbeddingRWKV WKV 算子")?;
        let mut builder = Session::builder()
            .context("无法初始化 ONNX Runtime")?
            .with_operators(operators)
            .context("无法配置 EmbeddingRWKV WKV 算子")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("无法配置 ONNX 图优化")?;
        if backend == RuntimeBackend::Cuda {
            builder = builder
                .with_execution_providers([CUDAExecutionProvider::default().build()])
                .context("无法启用 NVIDIA CUDA Execution Provider")?;
        }
        builder
            .commit_from_file(model_path)
            .with_context(|| format!("无法加载 EmbeddingRWKV 模型 {}", model_path.display()))
    }
}

impl RwkvModel {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let tokenized = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                let mut tokens = self.tokenizer.encode(text);
                tokens.push(EOS_TOKEN_ID);
                (index, tokens)
            })
            .collect::<Vec<_>>();
        let batches = plan_token_batches(&tokenized);
        let total_tokens = tokenized
            .iter()
            .map(|(_, tokens)| tokens.len())
            .sum::<usize>();
        let padded_tokens = batches
            .iter()
            .map(|batch| {
                batch.len()
                    * batch
                        .iter()
                        .map(|index| tokenized[*index].1.len())
                        .max()
                        .unwrap_or(0)
            })
            .sum::<usize>();
        tracing::debug!(
            documents = texts.len(),
            batches = batches.len(),
            total_tokens,
            padded_tokens,
            "planned EmbeddingRWKV token batches"
        );

        let mut vectors = vec![None; texts.len()];
        for batch in batches {
            let max_length = batch
                .iter()
                .map(|index| tokenized[*index].1.len())
                .max()
                .unwrap_or(1);
            let mut flattened = Vec::with_capacity(batch.len() * max_length);
            for index in &batch {
                let tokens = &tokenized[*index].1;
                flattened.extend(std::iter::repeat_n(0, max_length - tokens.len()));
                flattened.extend_from_slice(tokens);
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
            for (index, values) in batch.iter().zip(values.chunks_exact(EMBEDDING_DIMENSION)) {
                let mut vector = values.to_vec();
                normalize(&mut vector);
                vectors[tokenized[*index].0] = Some(vector);
            }
        }
        vectors
            .into_iter()
            .map(|vector| vector.context("EmbeddingRWKV 批处理结果缺失"))
            .collect()
    }
}

fn plan_token_batches(tokenized: &[(usize, Vec<i64>)]) -> Vec<Vec<usize>> {
    let mut order = (0..tokenized.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|index| tokenized[*index].1.len());
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_max_length = 0usize;
    for index in order {
        let length = tokenized[index].1.len();
        let next_max_length = current_max_length.max(length);
        let exceeds_batch = current.len() >= INFERENCE_MAX_BATCH_SIZE;
        let exceeds_tokens = !current.is_empty()
            && next_max_length * (current.len() + 1) > INFERENCE_MAX_PADDED_TOKENS;
        if exceeds_batch || exceeds_tokens {
            batches.push(std::mem::take(&mut current));
            current_max_length = 0;
        }
        current_max_length = current_max_length.max(length);
        current.push(index);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
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

fn configure_onnx_runtime() -> RuntimeBackend {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return RuntimeBackend::Cpu;
    }
    let Some(executable_dir) = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(PathBuf::from))
    else {
        return RuntimeBackend::Cpu;
    };
    let cuda_runtime_path = executable_dir.join("onnxruntime-cuda.dll");
    let use_cuda = cuda_runtime_path.is_file() && nvidia_cuda_available();
    let runtime_path = if use_cuda {
        prepare_cuda_library_search_path(&executable_dir);
        cuda_runtime_path
    } else {
        executable_dir.join("onnxruntime.dll")
    };
    if runtime_path.is_file() {
        std::env::set_var("ORT_DYLIB_PATH", runtime_path);
    }
    if use_cuda {
        RuntimeBackend::Cuda
    } else {
        RuntimeBackend::Cpu
    }
}

fn prepare_cuda_library_search_path(executable_dir: &std::path::Path) {
    let mut paths = vec![
        executable_dir.to_path_buf(),
        executable_dir.join("cuda-runtime"),
        executable_dir.join("cudnn-runtime"),
    ];
    if let Some(cuda_path) = std::env::var_os("CUDA_PATH") {
        paths.push(PathBuf::from(cuda_path).join("bin"));
    }
    if let Some(cudnn_path) = std::env::var_os("CUDNN_PATH") {
        paths.push(PathBuf::from(cudnn_path).join("bin"));
    }
    let existing = std::env::var_os("PATH").unwrap_or_default();
    let joined = std::env::join_paths(paths.into_iter().chain(std::env::split_paths(&existing)))
        .unwrap_or(existing);
    std::env::set_var("PATH", joined);
}

#[cfg(windows)]
fn nvidia_cuda_available() -> bool {
    type CuInit = unsafe extern "system" fn(u32) -> i32;
    type CuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> i32;
    unsafe {
        let Ok(library) = libloading::Library::new("nvcuda.dll") else {
            return false;
        };
        let Ok(cu_init) = library.get::<CuInit>(b"cuInit\0") else {
            return false;
        };
        let Ok(cu_device_get_count) = library.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0") else {
            return false;
        };
        if cu_init(0) != 0 {
            return false;
        }
        let mut device_count = 0;
        cu_device_get_count(&mut device_count) == 0 && device_count > 0
    }
}

#[cfg(not(windows))]
fn nvidia_cuda_available() -> bool {
    false
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

    #[test]
    fn token_batches_group_similar_lengths_and_preserve_all_inputs() {
        let tokenized = vec![
            (0, vec![0; 3_000]),
            (1, vec![0; 100]),
            (2, vec![0; 120]),
            (3, vec![0; 140]),
            (4, vec![0; 160]),
        ];
        let batches = plan_token_batches(&tokenized);
        let mut indexes = batches.iter().flatten().copied().collect::<Vec<_>>();
        indexes.sort_unstable();
        assert_eq!(indexes, (0..tokenized.len()).collect::<Vec<_>>());
        assert!(batches
            .iter()
            .all(|batch| batch.len() <= INFERENCE_MAX_BATCH_SIZE));
        assert_eq!(batches.last(), Some(&vec![0]));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_wkv_matches_scalar_kernel() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let token_count = 3;
        let value_count = token_count * EMBEDDING_DIMENSION;
        let values = |scale: f32, offset: usize| {
            (0..value_count)
                .map(|index| ((index + offset) % 29 + 1) as f32 * scale)
                .collect::<Vec<_>>()
        };
        let receptance = values(0.003, 1);
        let decay = (0..value_count)
            .map(|index| 0.97 - (index % 7) as f32 * 0.001)
            .collect::<Vec<_>>();
        let key = values(0.002, 3);
        let value = values(0.0025, 5);
        let in_context_key = values(0.0015, 7);
        let in_context_value = values(0.001, 11);
        let mut scalar = vec![0.0; value_count];
        let mut avx2 = vec![0.0; value_count];
        let state_size = RWKV_HEAD_COUNT * RWKV_HEAD_SIZE * RWKV_HEAD_SIZE;
        rwkv7_forward_batch_scalar(
            token_count,
            &receptance,
            &decay,
            &key,
            &value,
            &in_context_key,
            &in_context_value,
            &mut scalar,
            state_size,
        );
        unsafe {
            rwkv7_forward_batch_avx2(
                token_count,
                &receptance,
                &decay,
                &key,
                &value,
                &in_context_key,
                &in_context_value,
                &mut avx2,
                state_size,
            );
        }
        let max_difference = scalar
            .iter()
            .zip(&avx2)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_difference < 1e-5, "max difference: {max_difference}");
    }
}
