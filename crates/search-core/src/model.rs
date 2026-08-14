use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistedIndex {
    pub directories: Vec<String>,
    pub documents: Vec<DocumentRecord>,
    pub failures: Vec<IndexFailure>,
    pub last_indexed: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: String,
    pub path: String,
    pub name: String,
    pub extension: String,
    pub modified_ms: u64,
    pub size: u64,
    pub chunks: Vec<ChunkRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub text: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub terms: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexFailure {
    pub path: String,
    pub reason: String,
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

#[derive(Debug, Serialize)]
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
pub struct ServiceStats {
    pub status: String,
    pub document_count: usize,
    pub chunk_count: usize,
    pub failed_count: usize,
    pub processed_files: usize,
    pub total_files: usize,
    pub current_file: Option<String>,
    pub directories: Vec<String>,
    pub last_indexed: Option<String>,
}
