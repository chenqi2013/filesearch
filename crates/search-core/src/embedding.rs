use anyhow::{Context, Result};
use ort::execution_providers::CUDAExecutionProvider;
use ort::memory::Allocator;
use ort::session::{
    builder::GraphOptimizationLevel,
    run_options::{OutputSelector, RunOptions},
    Session, SessionInputValue,
};
use ort::value::Tensor;
use parking_lot::{Mutex, RwLock};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use tokenizers::{Tokenizer, TruncationParams};

pub const EMBEDDING_DIMENSION: usize = 1024;
pub const EMBEDDING_PROFILE: &str = "qwen3-embedding-0.6b-int8-document-chunks-v1";
const MODEL_NAME: &str = "Qwen3-Embedding-0.6B INT8";
const QUERY_INSTRUCTION: &str =
    "Given a user query, retrieve relevant passages from local documents that answer the query";
const MAX_SEQUENCE_LENGTH: usize = 512;
const INFERENCE_MAX_BATCH_SIZE: usize = 8;
const INFERENCE_MAX_PADDED_TOKENS: usize = 2_048;
const PAD_TOKEN_ID: i64 = 151_643;
const QWEN_LAYER_COUNT: usize = 28;
const QWEN_KV_HEAD_COUNT: usize = 8;
const QWEN_HEAD_SIZE: usize = 128;

struct QwenModel {
    session: Session,
    tokenizer: Tokenizer,
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
            Self::Cuda => "NVIDIA CUDA",
        }
    }
}

enum ModelState {
    Uninitialized,
    Ready(Box<QwenModel>),
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
        let query = format!("Instruct: {QUERY_INSTRUCTION}\nQuery:{query}");
        self.embed_batch(&[query.clone()])
            .into_iter()
            .next()
            .unwrap_or_else(|| fallback_embed(query.as_str()))
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

    fn load_model(&self) -> Result<QwenModel> {
        let model_path = self.model_dir.join("model_int8.onnx");
        let tokenizer_path = self.model_dir.join("tokenizer.json");
        let tokenizer = load_qwen_tokenizer(&tokenizer_path)?;
        if self.preferred_backend == RuntimeBackend::Cuda {
            match self.build_session(&model_path, RuntimeBackend::Cuda) {
                Ok(session) => {
                    return Ok(QwenModel {
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
        Ok(QwenModel {
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
        let thread_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(1, 8);
        let mut builder = Session::builder()
            .context("无法初始化 ONNX Runtime")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("无法配置 ONNX 图优化")?
            .with_intra_threads(thread_count)
            .context("无法配置 Qwen CPU 推理线程")?;
        if backend == RuntimeBackend::Cuda {
            builder = builder
                .with_execution_providers([CUDAExecutionProvider::default().build()])
                .context("无法启用 NVIDIA CUDA Execution Provider")?;
        }
        builder
            .commit_from_file(model_path)
            .with_context(|| format!("无法加载 Qwen Embedding 模型 {}", model_path.display()))
    }
}

impl QwenModel {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| anyhow::anyhow!("Qwen tokenizer 编码失败: {error}"))?;
        let tokenized = encodings
            .iter()
            .enumerate()
            .map(|(index, encoding)| {
                (
                    index,
                    encoding
                        .get_ids()
                        .iter()
                        .map(|token| i64::from(*token))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let batches = plan_token_batches(&tokenized);
        let mut vectors = vec![None; texts.len()];

        for batch in batches {
            let max_length = batch
                .iter()
                .map(|index| tokenized[*index].1.len())
                .max()
                .unwrap_or(1);
            let mut input_ids = Vec::with_capacity(batch.len() * max_length);
            let mut attention_mask = Vec::with_capacity(batch.len() * max_length);
            let mut position_ids = Vec::with_capacity(batch.len() * max_length);
            for index in &batch {
                let tokens = &tokenized[*index].1;
                let padding = max_length - tokens.len();
                input_ids.extend(std::iter::repeat_n(PAD_TOKEN_ID, padding));
                input_ids.extend_from_slice(tokens);
                attention_mask.extend(std::iter::repeat_n(0_i64, padding));
                attention_mask.extend(std::iter::repeat_n(1_i64, tokens.len()));
                position_ids.extend(std::iter::repeat_n(0_i64, padding));
                position_ids.extend((0..tokens.len()).map(|position| position as i64));
            }

            let batch_size = batch.len();
            let mut inputs: Vec<(String, SessionInputValue<'_>)> = vec![
                (
                    "input_ids".to_owned(),
                    Tensor::<i64>::from_array(([batch_size, max_length], input_ids))?.into(),
                ),
                (
                    "attention_mask".to_owned(),
                    Tensor::<i64>::from_array(([batch_size, max_length], attention_mask))?.into(),
                ),
                (
                    "position_ids".to_owned(),
                    Tensor::<i64>::from_array(([batch_size, max_length], position_ids))?.into(),
                ),
            ];
            for layer in 0..QWEN_LAYER_COUNT {
                let shape = [batch_size, QWEN_KV_HEAD_COUNT, 0, QWEN_HEAD_SIZE];
                inputs.push((
                    format!("past_key_values.{layer}.key"),
                    Tensor::<f32>::new(&Allocator::default(), shape)?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{layer}.value"),
                    Tensor::<f32>::new(&Allocator::default(), shape)?.into(),
                ));
            }
            let run_options = RunOptions::new()?
                .with_outputs(OutputSelector::no_default().with("last_hidden_state"));
            let outputs = self
                .session
                .run_with_options(inputs, &run_options)
                .context("Qwen Embedding ONNX 推理失败")?;
            let (shape, values) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .context("Qwen Embedding 输出格式无效")?;
            if shape.as_ref()
                != [
                    batch_size as i64,
                    max_length as i64,
                    EMBEDDING_DIMENSION as i64,
                ]
            {
                anyhow::bail!("Qwen Embedding 输出维度异常: {shape:?}");
            }
            let sequence_stride = max_length * EMBEDDING_DIMENSION;
            let last_token_offset = (max_length - 1) * EMBEDDING_DIMENSION;
            for (batch_position, index) in batch.iter().enumerate() {
                let start = batch_position * sequence_stride + last_token_offset;
                let mut vector = values[start..start + EMBEDDING_DIMENSION].to_vec();
                normalize(&mut vector);
                vectors[tokenized[*index].0] = Some(vector);
            }
        }

        vectors
            .into_iter()
            .map(|vector| vector.context("Qwen Embedding 批处理结果缺失"))
            .collect()
    }
}

fn load_qwen_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|error| anyhow::anyhow!("无法读取 Qwen tokenizer {}: {error}", path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_SEQUENCE_LENGTH,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!("无法配置 Qwen tokenizer 截断: {error}"))?;
    Ok(tokenizer)
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
    use serde::Deserialize;
    use std::time::Instant;

    #[derive(Deserialize)]
    struct RetrievalDocument {
        id: String,
        text: String,
    }

    #[derive(Deserialize)]
    struct RetrievalQuery {
        text: String,
        relevant: String,
    }

    #[derive(Deserialize)]
    struct RetrievalDataset {
        documents: Vec<RetrievalDocument>,
        queries: Vec<RetrievalQuery>,
    }

    fn retrieval_dataset() -> RetrievalDataset {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.codex-tmp/semantic-retrieval-eval.json");
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    fn report_retrieval_metrics(
        model: &str,
        dataset: &RetrievalDataset,
        document_vectors: &[Vec<f32>],
        query_vectors: &[Vec<f32>],
    ) {
        let mut top1 = 0usize;
        let mut recall3 = 0usize;
        let mut reciprocal_rank3 = 0.0f32;
        let mut failures = Vec::new();
        for (query, query_vector) in dataset.queries.iter().zip(query_vectors) {
            let mut ranking = dataset
                .documents
                .iter()
                .zip(document_vectors)
                .map(|(document, vector)| (document.id.as_str(), cosine(query_vector, vector)))
                .collect::<Vec<_>>();
            ranking.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            let rank = ranking
                .iter()
                .position(|(id, _)| *id == query.relevant)
                .map(|index| index + 1)
                .unwrap();
            top1 += usize::from(rank == 1);
            recall3 += usize::from(rank <= 3);
            if rank <= 3 {
                reciprocal_rank3 += 1.0 / rank as f32;
            }
            if rank != 1 {
                failures.push(format!(
                    "query={:?} expected={} rank={} top3={:?}",
                    query.text,
                    query.relevant,
                    rank,
                    ranking.iter().take(3).collect::<Vec<_>>()
                ));
            }
        }
        let count = dataset.queries.len() as f32;
        eprintln!(
            "{model} retrieval: queries={}, top1={:.1}%, recall@3={:.1}%, mrr@3={:.3}, failures={}",
            dataset.queries.len(),
            top1 as f32 * 100.0 / count,
            recall3 as f32 * 100.0 / count,
            reciprocal_rank3 / count,
            failures.len()
        );
        for failure in failures {
            eprintln!("{model} {failure}");
        }
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
            (0, vec![0; 500]),
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
    }

    #[test]
    fn qwen_tokenizer_adds_eos_without_early_padding() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/qwen3-embedding-0.6b/tokenizer.json");
        let tokenizer = load_qwen_tokenizer(&path).unwrap();
        let encodings = tokenizer
            .encode_batch(vec!["短文本", "这是稍微长一些的文本"], true)
            .unwrap();
        assert!(encodings[0].len() < encodings[1].len());
        assert_eq!(encodings[0].get_ids().last(), Some(&(PAD_TOKEN_ID as u32)));
        assert_eq!(encodings[1].get_ids().last(), Some(&(PAD_TOKEN_ID as u32)));
        assert!(encodings[0]
            .get_attention_mask()
            .iter()
            .all(|value| *value == 1));
    }

    #[test]
    #[ignore = "loads the bundled 600 MB Qwen model"]
    fn qwen_model_produces_relevant_normalized_embeddings() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/qwen3-embedding-0.6b");
        let engine = EmbeddingEngine::new(model_dir);
        let started = Instant::now();
        let mut model = engine.load_model().unwrap();
        let query_text = format!(
            "Instruct: {QUERY_INSTRUCTION}\nQuery:{}",
            "系统如何防止网络攻击"
        );
        let query = model.embed(&[query_text]).unwrap().remove(0);
        let load_and_query = started.elapsed();
        let started = Instant::now();
        let passages = model
            .embed(&[
                "系统采用防火墙、身份认证和入侵检测来保障网络安全。".to_owned(),
                "员工出差的交通费用按财务制度报销。".to_owned(),
                "The security platform detects and blocks cyber attacks.".to_owned(),
            ])
            .unwrap();
        let passage_time = started.elapsed();
        assert_eq!(query.len(), EMBEDDING_DIMENSION);
        assert!((cosine(&query, &query) - 1.0).abs() < 0.001);
        assert!(cosine(&query, &passages[0]) > cosine(&query, &passages[1]));
        assert!(cosine(&query, &passages[2]) > cosine(&query, &passages[1]));
        eprintln!(
            "Qwen backend={}, first_query={load_and_query:?}, passages={passage_time:?}, scores={:?}",
            model.backend.code(),
            passages
                .iter()
                .map(|passage| cosine(&query, passage))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "benchmarks the bundled 600 MB Qwen model"]
    fn qwen_model_benchmarks_indexing_batch() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/qwen3-embedding-0.6b");
        let engine = EmbeddingEngine::new(model_dir);
        let load_started = Instant::now();
        let mut model = engine.load_model().unwrap();
        let load_elapsed = load_started.elapsed();
        let texts = (0..16)
            .map(|index| {
                let body = "本地文档智能搜索需要支持中文语义检索、关键词匹配和索引性能统计。"
                    .repeat(20)
                    .chars()
                    .take(480)
                    .collect::<String>();
                format!(
                    "需求文档-{index}.docx\n{body}",
                )
            })
            .collect::<Vec<_>>();
        let token_lengths = model
            .tokenizer
            .encode_batch(texts.clone(), true)
            .unwrap()
            .iter()
            .map(|encoding| encoding.len())
            .collect::<Vec<_>>();

        assert_eq!(model.embed(&texts).unwrap().len(), texts.len());
        let elapsed = (0..3)
            .map(|_| {
                let started = Instant::now();
                assert_eq!(model.embed(&texts).unwrap().len(), texts.len());
                started.elapsed()
            })
            .collect::<Vec<_>>();
        eprintln!(
            "Qwen indexing batch: backend={}, load={load_elapsed:?}, texts={}, token_lengths={token_lengths:?}, elapsed={elapsed:?}",
            model.backend.code(),
            texts.len()
        );
    }

    #[test]
    #[ignore = "benchmarks the previous multilingual E5 model"]
    fn e5_model_benchmarks_indexing_batch() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release/models/multilingual-e5-small");
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let load_started = Instant::now();
        let mut session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(8)
            .unwrap()
            .commit_from_file(model_dir.join("onnx/model.onnx"))
            .unwrap();
        let load_elapsed = load_started.elapsed();
        let texts = (0..16)
            .map(|index| {
                let body = "本地文档智能搜索需要支持中文语义检索、关键词匹配和索引性能统计。"
                    .repeat(20)
                    .chars()
                    .take(480)
                    .collect::<String>();
                format!("passage: 需求文档-{index}.docx\n{body}")
            })
            .collect::<Vec<_>>();
        let encodings = tokenizer.encode_batch(texts, true).unwrap();
        let max_length = encodings.iter().map(|encoding| encoding.len()).max().unwrap();
        let token_lengths = encodings
            .iter()
            .map(|encoding| encoding.len())
            .collect::<Vec<_>>();
        let run = |session: &mut Session| {
            let mut input_ids = Vec::with_capacity(encodings.len() * max_length);
            let mut attention_mask = Vec::with_capacity(encodings.len() * max_length);
            for encoding in &encodings {
                let ids = encoding.get_ids();
                input_ids.extend(ids.iter().map(|token| i64::from(*token)));
                input_ids.extend(std::iter::repeat_n(0_i64, max_length - ids.len()));
                attention_mask.extend(std::iter::repeat_n(1_i64, ids.len()));
                attention_mask.extend(std::iter::repeat_n(0_i64, max_length - ids.len()));
            }
            let token_type_ids = vec![0_i64; encodings.len() * max_length];
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => Tensor::<i64>::from_array(([encodings.len(), max_length], input_ids)).unwrap(),
                    "attention_mask" => Tensor::<i64>::from_array(([encodings.len(), max_length], attention_mask)).unwrap(),
                    "token_type_ids" => Tensor::<i64>::from_array(([encodings.len(), max_length], token_type_ids)).unwrap(),
                ])
                .unwrap();
            let (shape, _) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .unwrap();
            assert_eq!(shape.as_ref(), [16, max_length as i64, 384]);
        };

        run(&mut session);
        let elapsed = (0..3)
            .map(|_| {
                let started = Instant::now();
                run(&mut session);
                started.elapsed()
            })
            .collect::<Vec<_>>();
        eprintln!(
            "E5 indexing batch: load={load_elapsed:?}, texts={}, token_lengths={token_lengths:?}, elapsed={elapsed:?}",
            encodings.len()
        );
    }

    #[test]
    #[ignore = "evaluates bundled Qwen and previous E5 models"]
    fn qwen_and_e5_retrieval_quality() {
        let dataset = retrieval_dataset();
        let qwen_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/qwen3-embedding-0.6b");
        let qwen_engine = EmbeddingEngine::new(qwen_dir);
        let mut qwen = qwen_engine.load_model().unwrap();
        let qwen_documents = qwen
            .embed(
                &dataset
                    .documents
                    .iter()
                    .map(|document| document.text.clone())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let qwen_queries = qwen
            .embed(
                &dataset
                    .queries
                    .iter()
                    .map(|query| format!("Instruct: {QUERY_INSTRUCTION}\nQuery:{}", query.text))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        report_retrieval_metrics(
            "Qwen3-Embedding-0.6B INT8",
            &dataset,
            &qwen_documents,
            &qwen_queries,
        );

        let e5_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/release/models/multilingual-e5-small");
        let tokenizer = Tokenizer::from_file(e5_dir.join("tokenizer.json")).unwrap();
        let mut session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(8)
            .unwrap()
            .commit_from_file(e5_dir.join("onnx/model.onnx"))
            .unwrap();
        let mut embed_e5 = |texts: Vec<String>| {
            let encodings = tokenizer.encode_batch(texts, true).unwrap();
            let max_length = encodings.iter().map(|encoding| encoding.len()).max().unwrap();
            let mut input_ids = Vec::with_capacity(encodings.len() * max_length);
            let mut attention_mask = Vec::with_capacity(encodings.len() * max_length);
            for encoding in &encodings {
                let ids = encoding.get_ids();
                input_ids.extend(ids.iter().map(|token| i64::from(*token)));
                input_ids.extend(std::iter::repeat_n(0_i64, max_length - ids.len()));
                attention_mask.extend(std::iter::repeat_n(1_i64, ids.len()));
                attention_mask.extend(std::iter::repeat_n(0_i64, max_length - ids.len()));
            }
            let token_type_ids = vec![0_i64; encodings.len() * max_length];
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => Tensor::<i64>::from_array(([encodings.len(), max_length], input_ids)).unwrap(),
                    "attention_mask" => Tensor::<i64>::from_array(([encodings.len(), max_length], attention_mask.clone())).unwrap(),
                    "token_type_ids" => Tensor::<i64>::from_array(([encodings.len(), max_length], token_type_ids)).unwrap(),
                ])
                .unwrap();
            let (_, values) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .unwrap();
            values
                .chunks_exact(max_length * 384)
                .zip(attention_mask.chunks_exact(max_length))
                .map(|(tokens, mask)| {
                    let mut vector = vec![0.0f32; 384];
                    let mut count = 0.0f32;
                    for (token, included) in tokens.chunks_exact(384).zip(mask) {
                        if *included == 0 {
                            continue;
                        }
                        count += 1.0;
                        for (target, value) in vector.iter_mut().zip(token) {
                            *target += *value;
                        }
                    }
                    for value in &mut vector {
                        *value /= count;
                    }
                    normalize(&mut vector);
                    vector
                })
                .collect::<Vec<_>>()
        };
        let e5_documents = embed_e5(
            dataset
                .documents
                .iter()
                .map(|document| format!("passage: {}", document.text))
                .collect(),
        );
        let e5_queries = embed_e5(
            dataset
                .queries
                .iter()
                .map(|query| format!("query: {}", query.text))
                .collect(),
        );
        report_retrieval_metrics(
            "multilingual-e5-small",
            &dataset,
            &e5_documents,
            &e5_queries,
        );
    }
}
