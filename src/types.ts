export type SearchMode = "hybrid" | "keyword" | "semantic";

export interface ServiceStats {
  status: "ready" | "indexing";
  document_count: number;
  chunk_count: number;
  failed_count: number;
  processed_files: number;
  total_files: number;
  current_file?: string;
  index_stage?: "ready" | "starting" | "scanning" | "checking" | "parsing" | "embedding" | "storage" | "text_index" | "committing";
  index_stage_elapsed_ms?: number;
  index_total_elapsed_ms?: number;
  index_scan_ms?: number;
  index_check_ms?: number;
  index_parse_ms?: number;
  index_embedding_ms?: number;
  index_storage_ms?: number;
  index_text_index_ms?: number;
  directories: string[];
  last_indexed?: string;
  storage_backend?: string;
  embedding_model?: string;
  embedding_backend?: "cpu" | "cuda" | "fallback";
  watcher_status?: string;
}

export interface SearchResult {
  id: string;
  path: string;
  name: string;
  extension: string;
  modified_ms: number;
  size: number;
  snippet: string;
  score: number;
}

export interface SearchResponse {
  query: string;
  total: number;
  elapsed_ms: number;
  results: SearchResult[];
}

export interface IndexFailure {
  path: string;
  category?: "encrypted" | "permission" | "offline" | "corrupt" | "parse" | "internal";
  reason: string;
}

export interface IndexedDocument {
  id: string;
  path: string;
  name: string;
  extension: string;
  modified_ms: number;
  size: number;
}

export interface IndexedChunk {
  id: number;
  document_id: string;
  document_name: string;
  document_path: string;
  position: number;
  text: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}
