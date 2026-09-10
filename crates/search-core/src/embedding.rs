use crate::rwkv::RwkvModel;
pub use crate::rwkv::EMBEDDING_DIMENSION;
use anyhow::{Context, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
const MODEL_NAME: &str = "EmbeddingRWKV Tiny";
pub const EMBEDDING_PROFILE: &str = "rwkv7-tiny-corrected-unpadded-document-chunks-v3";
const MAX_SEQUENCE_LENGTH: usize = 1024;
const INFERENCE_MAX_BATCH_SIZE: usize = 4;
const INFERENCE_MAX_PADDED_TOKENS: usize = 2_048;
const EOS_TOKEN_ID: i64 = 65535;

struct LocalRwkvModel {
    inference: RwkvModel,
    tokenizer: RwkvTokenizer,
    backend: RuntimeBackend,
}

#[derive(Default)]
struct TokenNode {
    children: HashMap<u8, usize>,
    token_id: Option<i64>,
}

struct RwkvTokenizer {
    nodes: Vec<TokenNode>,
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
            Self::Cuda => "NVIDIA CUDA（WKV 使用 CPU）",
        }
    }
}

enum ModelState {
    Uninitialized,
    Ready(Box<LocalRwkvModel>),
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

    fn load_model(&self) -> Result<LocalRwkvModel> {
        let model_path = self.model_dir.join("model.onnx");
        let tokenizer = RwkvTokenizer::load(&self.model_dir.join("rwkv_vocab.bin"))?;
        if self.preferred_backend == RuntimeBackend::Cuda {
            match RwkvModel::load_with_cuda(&model_path) {
                Ok(inference) => {
                    return Ok(LocalRwkvModel {
                        inference,
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
        Ok(LocalRwkvModel {
            inference: RwkvModel::load(&model_path)?,
            tokenizer,
            backend: RuntimeBackend::Cpu,
        })
    }
}

impl LocalRwkvModel {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let tokenized = texts
            .iter()
            .enumerate()
            .map(|(index, text)| (index, self.tokenizer.encode_with_eos(text)))
            .collect::<Vec<_>>();
        let batches = plan_token_batches(&tokenized);
        let mut vectors = vec![None; texts.len()];
        for batch in batches {
            let inputs = batch
                .iter()
                .map(|index| tokenized[*index].1.clone())
                .collect::<Vec<_>>();
            let outputs = self.inference.embed_tokens(&inputs)?;
            for (index, vector) in batch.into_iter().zip(outputs) {
                vectors[tokenized[index].0] = Some(vector);
            }
        }
        vectors
            .into_iter()
            .map(|vector| vector.context("EmbeddingRWKV batch output missing"))
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
        let needs_padding = !current.is_empty() && length != current_max_length;
        if exceeds_batch || exceeds_tokens || needs_padding {
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
        anyhow::ensure!(count == 65536, "Invalid RWKV vocabulary size");
        anyhow::ensure!(
            (0..=255_u8).all(|byte| tokenizer.nodes[0]
                .children
                .get(&byte)
                .is_some_and(|index| tokenizer.nodes[*index].token_id.is_some())),
            "RWKV vocabulary must cover every byte"
        );
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
        while offset < bytes.len() && tokens.len() < MAX_SEQUENCE_LENGTH - 1 {
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

impl RwkvTokenizer {
    fn encode_with_eos(&self, text: &str) -> Vec<i64> {
        let mut tokens = self.encode(text);
        tokens.push(EOS_TOKEN_ID);
        tokens
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
        assert!(batches.iter().all(|batch| batch
            .iter()
            .all(|index| tokenized[*index].1.len() == tokenized[batch[0]].1.len())));
    }

    #[test]
    fn token_batches_combine_equal_lengths_within_budget() {
        let tokenized = (0..9)
            .map(|index| (index, vec![1; 1024]))
            .collect::<Vec<_>>();
        let batches = plan_token_batches(&tokenized);
        assert_eq!(batches.len(), 5);
        assert!(batches
            .iter()
            .all(|batch| batch.len() * 1024 <= INFERENCE_MAX_PADDED_TOKENS));
    }

    fn rwkv_tokenizer() -> RwkvTokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/embedding-rwkv-tiny/rwkv_vocab.bin");
        RwkvTokenizer::load(&path).unwrap()
    }

    #[test]
    fn rwkv_tokenizer_matches_official_world_tokenizer() {
        let tokenizer = rwkv_tokenizer();
        assert_eq!(
            tokenizer.encode("本地文档搜索"),
            [13205, 11459, 13012, 13351, 12877, 15325]
        );
        assert_eq!(
            tokenizer.encode("EmbeddingRWKV Tiny test"),
            [33071, 25139, 1413, 1184, 29906, 32223]
        );
    }

    #[test]
    fn rwkv_tokenizer_reserves_single_eos_after_truncation() {
        let tokenizer = rwkv_tokenizer();
        for text in [
            "".to_owned(),
            "短文本".to_owned(),
            "本地文档搜索".repeat(1024),
        ] {
            let tokens = tokenizer.encode_with_eos(&text);
            assert!(tokens.len() <= MAX_SEQUENCE_LENGTH);
            assert_eq!(tokens.last(), Some(&EOS_TOKEN_ID));
            assert_eq!(
                tokens
                    .iter()
                    .filter(|token| **token == EOS_TOKEN_ID)
                    .count(),
                1
            );
        }
        assert_eq!(tokenizer.encode_with_eos("").len(), 1);
        assert_eq!(
            tokenizer
                .encode_with_eos(&"本地文档搜索".repeat(1024))
                .len(),
            MAX_SEQUENCE_LENGTH
        );
    }

    #[test]
    #[ignore = "loads the bundled RWKV model"]
    fn rwkv_model_produces_relevant_normalized_embeddings() {
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/embedding-rwkv-tiny");
        let engine = EmbeddingEngine::new(model_dir);
        let started = Instant::now();
        let mut model = engine.load_model().unwrap();
        let load_elapsed = started.elapsed();
        let texts = vec![
            "系统如何防止网络攻击".to_owned(),
            "员工出差的交通费用按财务制度报销。".to_owned(),
            "系统采用防火墙、身份认证和入侵检测来保障网络安全。".to_owned(),
            "The security platform detects and blocks cyber attacks.".to_owned(),
            "短文本".to_owned(),
        ];
        let started = Instant::now();
        let vectors = model.embed(&texts).unwrap();
        let embedding_elapsed = started.elapsed();
        for (text, vector) in texts.iter().zip(&vectors) {
            let single = model.embed(&[text.clone()]).unwrap().remove(0);
            assert_eq!(vector.len(), EMBEDDING_DIMENSION);
            assert!((cosine(vector, vector) - 1.0).abs() < 0.001);
            let similarity = cosine(vector, &single);
            eprintln!("RWKV batch/single cosine={similarity:.8}, text={text:?}");
            assert!(similarity > 0.9999, "batch/single cosine={similarity}");
        }
        assert!(cosine(&vectors[0], &vectors[2]) > cosine(&vectors[0], &vectors[1]));
        assert!(cosine(&vectors[0], &vectors[3]) > cosine(&vectors[0], &vectors[1]));
        *engine.state.lock() = ModelState::Ready(Box::new(model));
        assert!(cosine(&engine.embed_query(&texts[0]), &vectors[0]) > 0.9999);
        let repeated = engine.embed_passages(&vec![texts[0].clone(); 5]);
        assert!(repeated
            .iter()
            .all(|vector| cosine(vector, &vectors[0]) > 0.9999));
        eprintln!(
            "RWKV load={load_elapsed:?}, five_texts={embedding_elapsed:?}, backend={}",
            engine.backend()
        );
    }

    #[test]
    #[ignore = "evaluates the bundled RWKV model against the local retrieval dataset"]
    fn rwkv_retrieval_quality() {
        let dataset = retrieval_dataset();
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/models/embedding-rwkv-tiny");
        let engine = EmbeddingEngine::new(model_dir);
        let mut model = engine.load_model().unwrap();
        let documents = dataset
            .documents
            .iter()
            .map(|document| document.text.clone())
            .collect::<Vec<_>>();
        let queries = dataset
            .queries
            .iter()
            .map(|query| query.text.clone())
            .collect::<Vec<_>>();
        let document_vectors = model.embed(&documents).unwrap();
        let query_vectors = model.embed(&queries).unwrap();
        report_retrieval_metrics(MODEL_NAME, &dataset, &document_vectors, &query_vectors);
    }
}
