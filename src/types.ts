export type SearchMode = "hybrid" | "keyword" | "semantic";

export interface ServiceDiagnostics {
  state: "running" | "starting" | "recovering" | "recovery_failed" | "unresponsive" | "unavailable" | "stopped";
  pid: number | null;
  detail: string;
  last_exit: { timestamp_ms: number; code: number | null; reason: string } | null;
  log_path: string;
  log_tail: string;
  restart_attempts: number;
  build: string;
}

export interface ServiceStats {
  status: "ready" | "indexing" | "paused";
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
  embedding_windows_processed?: number;
  embedding_windows_total?: number;
  directories: string[];
  extensions: string[];
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
  duplicates?: IndexedDocument[];
}

export interface SearchResponse {
  query: string;
  total: number;
  elapsed_ms: number;
  results: SearchResult[];
}

export interface IndexFailure {
  path: string;
  category?: "dependency" | "encrypted" | "permission" | "offline" | "corrupt" | "timeout" | "parse" | "internal";
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

export interface FileIndexTiming {
  path: string;
  extension: string;
  size: number;
  status: "indexed" | "failed" | "skipped";
  check_ms: number;
  parse_ms: number;
  ocr_ms: number;
  embedding_ms: number;
  storage_ms: number;
  text_index_ms: number;
  total_ms: number;
  recorded_at: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}
