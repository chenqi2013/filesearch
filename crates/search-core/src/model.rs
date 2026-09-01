use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LegacyIndex {
    #[serde(default)]
    pub directories: Vec<String>,
    #[serde(default)]
    pub documents: Vec<LegacyDocument>,
    #[serde(default)]
    pub failures: Vec<IndexFailure>,
    pub last_indexed: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyDocument {
    pub id: String,
    pub path: String,
    pub name: String,
    pub extension: String,
    pub modified_ms: u64,
    pub size: u64,
    #[serde(default)]
    pub chunks: Vec<LegacyChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyChunk {
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct PreparedDocument {
    pub id: String,
    pub root: String,
    pub path: String,
    pub name: String,
    pub extension: String,
    pub modified_ms: u64,
    pub size: u64,
    pub chunks: Vec<String>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct StoredDocument {
    pub id: String,
    pub root: String,
    pub path: String,
    pub name: String,
    pub extension: String,
    pub modified_ms: u64,
    pub size: u64,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub id: u64,
    pub document_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexFailure {
    pub path: String,
    #[serde(default = "default_failure_category")]
    pub category: String,
    pub reason: String,
}

fn default_failure_category() -> String {
    "parse".to_owned()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Keyword,
    Semantic,
    #[default]
    Hybrid,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub mode: SearchMode,
    pub extension: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    20
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub total: usize,
    pub elapsed_ms: u128,
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub path: String,
    pub name: String,
    pub extension: String,
    pub modified_ms: u64,
    pub size: u64,
    pub snippet: String,
    pub score: f32,
}

#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    pub paths: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct IndexAccepted {
    pub accepted: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexedDocument {
    pub id: String,
    pub path: String,
    pub name: String,
    pub extension: String,
    pub modified_ms: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexedChunk {
    pub id: u64,
    pub document_id: String,
    pub document_name: String,
    pub document_path: String,
    pub position: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub total: usize,
    pub offset: usize,
    pub limit: usize,
}

#[derive(Debug, Deserialize, Default)]
pub struct PageRequest {
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_page_limit")]
    pub limit: usize,
}

impl PageRequest {
    pub fn bounded(self) -> Self {
        Self {
            offset: self.offset,
            limit: self.limit.clamp(1, 100),
        }
    }
}

fn default_page_limit() -> usize {
    50
}

#[derive(Debug, Clone, Serialize)]
pub struct ServiceStats {
    pub status: String,
    pub document_count: usize,
    pub chunk_count: usize,
    pub failed_count: usize,
    pub processed_files: usize,
    pub total_files: usize,
    pub current_file: Option<String>,
    pub index_stage: String,
    pub index_stage_elapsed_ms: u128,
    pub index_total_elapsed_ms: u128,
    pub index_scan_ms: u128,
    pub index_check_ms: u128,
    pub index_parse_ms: u128,
    pub index_embedding_ms: u128,
    pub index_storage_ms: u128,
    pub index_text_index_ms: u128,
    pub directories: Vec<String>,
    pub last_indexed: Option<String>,
    pub storage_backend: String,
    pub embedding_model: String,
    pub embedding_backend: String,
    pub watcher_status: String,
}
