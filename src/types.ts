export type SearchMode = "hybrid" | "keyword" | "semantic";

export interface ServiceStats {
  status: "ready" | "indexing";
  document_count: number;
  chunk_count: number;
  failed_count: number;
  processed_files: number;
  total_files: number;
  current_file?: string;
  directories: string[];
  last_indexed?: string;
  storage_backend?: string;
  embedding_model?: string;
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
