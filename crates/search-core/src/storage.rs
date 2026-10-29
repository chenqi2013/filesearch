use crate::model::{
    FileIndexTiming, IndexFailure, IndexedChunk, IndexedDocument, LegacyIndex, PreparedDocument,
    StoredChunk, StoredDocument,
};
use anyhow::{Context, Result};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Storage {
    connection: Mutex<Connection>,
    embeddings: RwLock<HashMap<String, CachedEmbedding>>,
    chunk_embeddings: RwLock<HashMap<u64, CachedChunkEmbedding>>,
    anchor_embeddings: RwLock<HashMap<u64, CachedChunkEmbedding>>,
    fingerprints: RwLock<HashMap<String, String>>,
}

struct CachedEmbedding {
    extension: String,
    vector: Vec<f32>,
}

struct CachedChunkEmbedding {
    chunk_id: u64,
    document_id: String,
    extension: String,
    vector: Vec<f32>,
    text: String,
}

pub struct SemanticChunkMatch {
    pub chunk_id: u64,
    pub document_id: String,
    pub text: String,
    pub score: f32,
    pub confirmation: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct StorageCounts {
    pub documents: usize,
    pub chunks: usize,
    pub failures: usize,
}

impl Storage {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(path)
            .with_context(|| format!("无法打开 SQLite 索引 {}", path.display()))?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;
             PRAGMA temp_store=MEMORY;
             PRAGMA busy_timeout=5000;
             CREATE TABLE IF NOT EXISTS settings (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS directories (
               path TEXT PRIMARY KEY
             );
             CREATE TABLE IF NOT EXISTS documents (
               id TEXT PRIMARY KEY,
               root TEXT NOT NULL,
               path TEXT NOT NULL UNIQUE,
               name TEXT NOT NULL,
               extension TEXT NOT NULL,
               modified_ms INTEGER NOT NULL,
               size INTEGER NOT NULL,
               embedding BLOB,
               indexed_at TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_documents_root ON documents(root);
             CREATE INDEX IF NOT EXISTS idx_documents_extension ON documents(extension);
             CREATE TABLE IF NOT EXISTS chunks (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
               position INTEGER NOT NULL,
               text TEXT NOT NULL,
               UNIQUE(document_id, position)
             );
             CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
             CREATE TABLE IF NOT EXISTS chunk_embeddings (
               chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
               document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
               embedding BLOB NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_document ON chunk_embeddings(document_id);
             CREATE TABLE IF NOT EXISTS semantic_passages (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
               document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
               text TEXT NOT NULL,
               embedding BLOB NOT NULL,
               UNIQUE(chunk_id, text)
             );
             CREATE INDEX IF NOT EXISTS idx_semantic_passages_document ON semantic_passages(document_id);
             CREATE TABLE IF NOT EXISTS semantic_anchors (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
               document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
               text TEXT NOT NULL,
               embedding BLOB NOT NULL,
               UNIQUE(chunk_id, text)
             );
             CREATE INDEX IF NOT EXISTS idx_semantic_anchors_document ON semantic_anchors(document_id);
             CREATE TABLE IF NOT EXISTS passage_vector_cache (
               cache_key TEXT PRIMARY KEY,
               embedding BLOB NOT NULL,
               used_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_passage_vector_cache_used ON passage_vector_cache(used_at);
             CREATE TABLE IF NOT EXISTS document_fingerprints (
               document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
               fingerprint TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS failures (
               path TEXT PRIMARY KEY,
               category TEXT NOT NULL,
               reason TEXT NOT NULL,
               updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS file_index_timings (
               path TEXT PRIMARY KEY,
               extension TEXT NOT NULL,
               size INTEGER NOT NULL,
               status TEXT NOT NULL,
               check_ms INTEGER NOT NULL,
               parse_ms INTEGER NOT NULL,
               ocr_ms INTEGER NOT NULL,
               embedding_ms INTEGER NOT NULL,
               storage_ms INTEGER NOT NULL,
               text_index_ms INTEGER NOT NULL,
               total_ms INTEGER NOT NULL,
               recorded_at TEXT NOT NULL
             );",
        )?;
        let has_ocr_column = {
            let mut statement = connection.prepare("PRAGMA table_info(file_index_timings)")?;
            let columns = statement.query_map([], |row| row.get::<_, String>(1))?;
            columns
                .collect::<rusqlite::Result<Vec<_>>>()?
                .iter()
                .any(|name| name == "ocr_ms")
        };
        if !has_ocr_column {
            connection.execute(
                "ALTER TABLE file_index_timings ADD COLUMN ocr_ms INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        let embeddings = load_embedding_cache(&connection)?;
        let chunk_embeddings = load_chunk_embedding_cache(&connection, false)?;
        let anchor_embeddings = load_chunk_embedding_cache(&connection, true)?;
        let fingerprints = {
            let mut statement =
                connection.prepare("SELECT document_id, fingerprint FROM document_fingerprints")?;
            let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
            rows.collect::<rusqlite::Result<HashMap<String, String>>>()?
        };
        Ok(Self {
            connection: Mutex::new(connection),
            embeddings: RwLock::new(embeddings),
            chunk_embeddings: RwLock::new(chunk_embeddings),
            anchor_embeddings: RwLock::new(anchor_embeddings),
            fingerprints: RwLock::new(fingerprints),
        })
    }

    pub fn migrate_legacy(&self, index_path: &Path) -> Result<bool> {
        if self.counts()?.documents > 0 || !index_path.exists() {
            return Ok(false);
        }
        let bytes = std::fs::read(index_path)?;
        let legacy: LegacyIndex =
            serde_json::from_slice(&bytes).context("旧版 index.json 无法解析")?;
        self.set_directories(&legacy.directories)?;
        for document in legacy.documents {
            let root = legacy
                .directories
                .iter()
                .filter(|root| document.path.starts_with(root.as_str()))
                .max_by_key(|root| root.len())
                .cloned()
                .unwrap_or_default();
            self.upsert_document(&PreparedDocument {
                id: document.id,
                root,
                path: document.path,
                name: document.name,
                extension: document.extension,
                modified_ms: document.modified_ms,
                size: document.size,
                chunks: document
                    .chunks
                    .into_iter()
                    .map(|chunk| chunk.text)
                    .collect(),
                embedding: Vec::new(),
            })?;
        }
        for failure in legacy.failures {
            self.record_failure(&failure)?;
        }
        if let Some(last_indexed) = legacy.last_indexed {
            self.set_setting("last_indexed", &last_indexed)?;
        }
        self.set_setting("legacy_migrated", "true")?;
        Ok(true)
    }

    pub fn counts(&self) -> Result<StorageCounts> {
        let connection = self.connection.lock();
        Ok(StorageCounts {
            documents: query_count(&connection, "SELECT COUNT(*) FROM documents")?,
            chunks: query_count(&connection, "SELECT COUNT(*) FROM chunks")?,
            failures: query_count(&connection, "SELECT COUNT(*) FROM failures")?,
        })
    }

    pub fn directories(&self) -> Result<Vec<String>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT path FROM directories ORDER BY path")?;
        let rows = statement.query_map([], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn extensions(&self) -> Result<Vec<String>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT DISTINCT extension FROM documents
             WHERE extension <> '' ORDER BY extension COLLATE NOCASE",
        )?;
        let rows = statement.query_map([], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn set_directories(&self, paths: &[String]) -> Result<()> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        transaction.execute("DELETE FROM directories", [])?;
        {
            let mut statement =
                transaction.prepare("INSERT OR IGNORE INTO directories(path) VALUES (?)")?;
            for path in paths {
                statement.execute([path])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn last_indexed(&self) -> Result<Option<String>> {
        self.setting("last_indexed")
    }

    pub fn set_last_indexed_now(&self) -> Result<()> {
        self.set_setting("last_indexed", &chrono::Utc::now().to_rfc3339())
    }

    pub fn list_documents(&self) -> Result<Vec<StoredDocument>> {
        self.query_documents(
            "SELECT id, root, path, name, extension, modified_ms, size, embedding FROM documents",
        )
    }

    pub fn list_documents_without_embeddings(&self) -> Result<Vec<StoredDocument>> {
        self.query_documents(
            "SELECT id, root, path, name, extension, modified_ms, size, NULL FROM documents",
        )
    }

    fn query_documents(&self, sql: &str) -> Result<Vec<StoredDocument>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(sql)?;
        let rows = statement.query_map([], document_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn list_document_paths(&self) -> Result<Vec<(String, String)>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT id, path FROM documents")?;
        let rows = statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn document_page(&self, offset: usize, limit: usize) -> Result<Vec<IndexedDocument>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, path, name, extension, modified_ms, size
             FROM documents ORDER BY name COLLATE NOCASE, path LIMIT ? OFFSET ?",
        )?;
        let rows = statement.query_map(params![limit as i64, offset as i64], |row| {
            Ok(IndexedDocument {
                id: row.get(0)?,
                path: row.get(1)?,
                name: row.get(2)?,
                extension: row.get(3)?,
                modified_ms: row.get::<_, i64>(4)? as u64,
                size: row.get::<_, i64>(5)? as u64,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn document_by_path(&self, path: &str) -> Result<Option<StoredDocument>> {
        let connection = self.connection.lock();
        connection
            .query_row(
                "SELECT id, root, path, name, extension, modified_ms, size, embedding
                 FROM documents WHERE path = ?",
                [path],
                document_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn upsert_document(&self, document: &PreparedDocument) -> Result<Vec<StoredChunk>> {
        Ok(self
            .upsert_documents(std::slice::from_ref(document))?
            .pop()
            .unwrap_or_default())
    }

    pub fn upsert_documents(
        &self,
        documents: &[PreparedDocument],
    ) -> Result<Vec<Vec<StoredChunk>>> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        let indexed_at = chrono::Utc::now().to_rfc3339();
        let mut all_chunks = Vec::with_capacity(documents.len());
        for document in documents {
            transaction.execute(
                "DELETE FROM documents WHERE id = ? OR path = ?",
                params![document.id, document.path],
            )?;
            let embedding =
                (!document.embedding.is_empty()).then(|| vector_to_blob(&document.embedding));
            transaction.execute(
                "INSERT INTO documents(id, root, path, name, extension, modified_ms, size, embedding, indexed_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    document.id,
                    document.root,
                    document.path,
                    document.name,
                    document.extension,
                    document.modified_ms as i64,
                    document.size as i64,
                    embedding,
                    indexed_at,
                ],
            )?;
            let mut chunks = Vec::with_capacity(document.chunks.len());
            {
                let mut statement = transaction
                    .prepare("INSERT INTO chunks(document_id, position, text) VALUES (?, ?, ?)")?;
                for (position, text) in document.chunks.iter().enumerate() {
                    statement.execute(params![document.id, position as i64, text])?;
                    chunks.push(StoredChunk {
                        id: transaction.last_insert_rowid() as u64,
                        document_id: document.id.clone(),
                        text: text.clone(),
                    });
                }
            }
            transaction.execute("DELETE FROM failures WHERE path = ?", [&document.path])?;
            transaction.execute(
                "INSERT INTO document_fingerprints(document_id, fingerprint) VALUES (?, ?)",
                params![document.id, text_fingerprint(&document.chunks)],
            )?;
            all_chunks.push(chunks);
        }
        transaction.commit()?;
        drop(connection);
        let document_ids = documents
            .iter()
            .map(|document| document.id.as_str())
            .collect::<std::collections::HashSet<_>>();
        self.chunk_embeddings
            .write()
            .retain(|_, embedding| !document_ids.contains(embedding.document_id.as_str()));
        self.anchor_embeddings
            .write()
            .retain(|_, embedding| !document_ids.contains(embedding.document_id.as_str()));
        let mut cache = self.embeddings.write();
        for document in documents {
            self.fingerprints
                .write()
                .insert(document.id.clone(), text_fingerprint(&document.chunks));
            if document.embedding.is_empty() {
                cache.remove(&document.id);
            } else {
                cache.insert(
                    document.id.clone(),
                    CachedEmbedding {
                        extension: document.extension.clone(),
                        vector: document.embedding.clone(),
                    },
                );
            }
        }
        Ok(all_chunks)
    }

    pub fn delete_document(&self, id: &str) -> Result<()> {
        self.connection
            .lock()
            .execute("DELETE FROM documents WHERE id = ?", [id])?;
        self.embeddings.write().remove(id);
        self.fingerprints.write().remove(id);
        self.chunk_embeddings
            .write()
            .retain(|_, embedding| embedding.document_id != id);
        self.anchor_embeddings
            .write()
            .retain(|_, embedding| embedding.document_id != id);
        Ok(())
    }

    pub fn semantic_search(
        &self,
        query: &[f32],
        extension: Option<&str>,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let cache = self.embeddings.read();
        let mut scores = cache
            .par_iter()
            .filter(|(_, embedding)| extension.is_none_or(|value| value == embedding.extension))
            .filter_map(|(document_id, embedding)| {
                let score = crate::embedding::cosine(query, &embedding.vector).max(0.0);
                (score > 0.05).then(|| (document_id.clone(), score))
            })
            .collect::<Vec<_>>();
        scores.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        scores.truncate(limit);
        scores
    }

    #[cfg(test)]
    pub fn semantic_chunk_search(
        &self,
        query: &[f32],
        extension: Option<&str>,
        limit: usize,
    ) -> Vec<(String, String, f32)> {
        self.semantic_chunk_matches(query, query, extension, limit)
            .into_iter()
            .map(|hit| (hit.document_id, hit.text, hit.score))
            .collect()
    }

    #[cfg(test)]
    pub fn semantic_chunk_matches(
        &self,
        query: &[f32],
        confirmation: &[f32],
        extension: Option<&str>,
        limit: usize,
    ) -> Vec<SemanticChunkMatch> {
        self.semantic_passage_matches(query, confirmation, extension, limit, 1, &[])
    }

    pub fn semantic_passage_matches(
        &self,
        query: &[f32],
        confirmation: &[f32],
        extension: Option<&str>,
        limit: usize,
        windows_per_document: usize,
        keyword_chunks: &[u64],
    ) -> Vec<SemanticChunkMatch> {
        let cache = self.chunk_embeddings.read();
        let scores = cache
            .par_iter()
            .filter(|(_, embedding)| extension.is_none_or(|value| value == embedding.extension))
            .map(|(passage_id, embedding)| {
                (
                    *passage_id,
                    crate::embedding::cosine(query, &embedding.vector).max(0.0),
                    crate::embedding::cosine(confirmation, &embedding.vector).max(0.0),
                )
            })
            .collect::<Vec<_>>();
        let compare_raw = |left: &(u64, f32, f32), right: &(u64, f32, f32)| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        };
        let mut grouped: HashMap<&str, Vec<_>> = HashMap::new();
        for score in scores {
            grouped
                .entry(cache[&score.0].document_id.as_str())
                .or_default()
                .push(score);
        }
        let mut documents = grouped
            .into_values()
            .map(|windows| {
                let best = *windows
                    .iter()
                    .min_by(|left, right| compare_raw(left, right))
                    .unwrap();
                (best, windows)
            })
            .collect::<Vec<_>>();
        let document_limit = limit.min(documents.len());
        if document_limit < documents.len() {
            documents.select_nth_unstable_by(document_limit, |left, right| {
                compare_raw(&left.0, &right.0)
            });
            documents.truncate(document_limit);
        }
        let keyword_chunks = keyword_chunks
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut selected = Vec::new();
        for (_, mut windows) in documents {
            let count = windows_per_document.clamp(1, 16).min(windows.len());
            let mut retained = std::collections::HashSet::new();
            if count < windows.len() {
                windows.select_nth_unstable_by(count, compare_raw);
            }
            retained.extend(windows[..count].iter().map(|window| window.0));
            if windows_per_document > 1 && count < windows.len() {
                windows.select_nth_unstable_by(count, |left, right| {
                    (right.1 * right.2)
                        .total_cmp(&(left.1 * left.2))
                        .then_with(|| left.0.cmp(&right.0))
                });
                retained.extend(windows[..count].iter().map(|window| window.0));
            }
            selected.extend(windows.into_iter().filter(|window| {
                retained.contains(&window.0) || keyword_chunks.contains(&cache[&window.0].chunk_id)
            }));
        }
        selected.sort_unstable_by(compare_raw);
        selected
            .into_iter()
            .map(|(passage_id, score, confirmation)| {
                let passage = &cache[&passage_id];
                SemanticChunkMatch {
                    chunk_id: passage.chunk_id,
                    document_id: passage.document_id.clone(),
                    text: passage.text.clone(),
                    score,
                    confirmation,
                }
            })
            .collect()
    }

    pub fn semantic_anchor_search(
        &self,
        query: &[f32],
        extension: Option<&str>,
    ) -> Vec<(String, String, f32, f32)> {
        let bodies = self.chunk_embeddings.read();
        let anchors = self.anchor_embeddings.read();
        if anchors.is_empty() {
            return Vec::new();
        }
        let anchored_chunks = anchors
            .values()
            .map(|anchor| anchor.chunk_id)
            .collect::<std::collections::HashSet<_>>();
        let mut evidence: HashMap<u64, (f32, &str)> = HashMap::new();
        for body in bodies.values().filter(|body| {
            anchored_chunks.contains(&body.chunk_id)
                && extension.is_none_or(|value| value == body.extension)
        }) {
            let score = crate::embedding::cosine(query, &body.vector).max(0.0);
            let entry = evidence.entry(body.chunk_id).or_insert((0.0, &body.text));
            if score > entry.0 {
                *entry = (score, &body.text);
            }
        }
        anchors
            .values()
            .filter(|anchor| extension.is_none_or(|value| value == anchor.extension))
            .filter(|anchor| crate::semantic_structure::is_descriptive_anchor(&anchor.text))
            .filter_map(|anchor| {
                let (body_score, text) = evidence.get(&anchor.chunk_id)?;
                Some((
                    anchor.document_id.clone(),
                    (*text).to_owned(),
                    crate::embedding::cosine(query, &anchor.vector).max(0.0),
                    *body_score,
                ))
            })
            .collect()
    }

    pub fn replace_chunk_embeddings(&self, embeddings: &[(u64, String, Vec<f32>)]) -> Result<()> {
        self.replace_semantic_vectors(embeddings, false)
    }

    pub fn replace_anchor_embeddings(&self, embeddings: &[(u64, String, Vec<f32>)]) -> Result<()> {
        self.replace_semantic_vectors(embeddings, true)
    }

    fn replace_semantic_vectors(
        &self,
        embeddings: &[(u64, String, Vec<f32>)],
        anchors: bool,
    ) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        let mut cached = Vec::with_capacity(embeddings.len());
        for (chunk_id, text, embedding) in embeddings {
            let Some((document_id, extension)) = transaction
                .query_row(
                    "SELECT c.document_id, d.extension
                     FROM chunks c JOIN documents d ON d.id = c.document_id
                     WHERE c.id = ?",
                    [*chunk_id as i64],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()?
            else {
                continue;
            };
            let table = if anchors {
                "semantic_anchors"
            } else {
                "semantic_passages"
            };
            transaction.execute(
                &format!("INSERT INTO {table}(chunk_id, document_id, text, embedding)
                 VALUES (?, ?, ?, ?) ON CONFLICT(chunk_id, text) DO UPDATE SET embedding=excluded.embedding"),
                params![*chunk_id as i64, &document_id, text, vector_to_blob(embedding)],
            )?;
            let passage_id: i64 = transaction.query_row(
                &format!("SELECT id FROM {table} WHERE chunk_id = ? AND text = ?"),
                params![*chunk_id as i64, text],
                |row| row.get(0),
            )?;
            cached.push((
                passage_id as u64,
                CachedChunkEmbedding {
                    chunk_id: *chunk_id,
                    document_id,
                    extension,
                    vector: embedding.clone(),
                    text: text.clone(),
                },
            ));
        }
        transaction.commit()?;
        drop(connection);
        let mut cache = if anchors {
            self.anchor_embeddings.write()
        } else {
            self.chunk_embeddings.write()
        };
        for (chunk_id, embedding) in cached {
            cache.insert(chunk_id, embedding);
        }
        Ok(())
    }

    pub fn cached_passage_vectors(&self, inputs: &[String]) -> Result<Vec<Option<Vec<f32>>>> {
        let connection = self.connection.lock();
        let mut statement = connection
            .prepare_cached("SELECT embedding FROM passage_vector_cache WHERE cache_key = ?")?;
        inputs
            .iter()
            .map(|input| {
                let blob = statement
                    .query_row([passage_cache_key(input)], |row| row.get::<_, Vec<u8>>(0))
                    .optional()?;
                Ok(blob.map(|bytes| blob_to_vector(&bytes)).filter(|vector| {
                    vector.len() == crate::embedding::EMBEDDING_DIMENSION
                        && vector.iter().all(|value| value.is_finite())
                }))
            })
            .collect()
    }

    pub fn cache_passage_vectors(&self, inputs: &[String], vectors: &[Vec<f32>]) -> Result<()> {
        anyhow::ensure!(
            inputs.len() == vectors.len(),
            "Missing passage cache vectors"
        );
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        {
            let mut statement = transaction.prepare_cached(
                "INSERT INTO passage_vector_cache(cache_key, embedding, used_at) VALUES (?, ?, ?)
                 ON CONFLICT(cache_key) DO UPDATE SET embedding=excluded.embedding, used_at=excluded.used_at",
            )?;
            let now = chrono::Utc::now().timestamp_millis();
            for (input, vector) in inputs.iter().zip(vectors) {
                statement.execute(params![
                    passage_cache_key(input),
                    vector_to_blob(vector),
                    now
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn prune_passage_cache(&self) -> Result<()> {
        let connection = self.connection.lock();
        let count: i64 =
            connection.query_row("SELECT COUNT(*) FROM passage_vector_cache", [], |row| {
                row.get(0)
            })?;
        if count <= 100_000 {
            return Ok(());
        }
        connection.execute(
            "DELETE FROM passage_vector_cache WHERE cache_key IN (
               SELECT cache_key FROM passage_vector_cache ORDER BY used_at DESC, cache_key
               LIMIT -1 OFFSET 100000
             )",
            [],
        )?;
        Ok(())
    }

    pub fn has_embedding(&self, document_id: &str) -> bool {
        self.embeddings.read().contains_key(document_id)
    }

    pub fn content_fingerprints(&self) -> HashMap<String, String> {
        self.fingerprints.read().clone()
    }

    pub fn update_document_embeddings(&self, documents: &[PreparedDocument]) -> Result<()> {
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        for document in documents {
            transaction.execute(
                "UPDATE documents SET embedding = ? WHERE id = ?",
                params![vector_to_blob(&document.embedding), document.id],
            )?;
        }
        transaction.commit()?;
        let mut cache = self.embeddings.write();
        for document in documents {
            cache.insert(
                document.id.clone(),
                CachedEmbedding {
                    extension: document.extension.clone(),
                    vector: document.embedding.clone(),
                },
            );
        }
        Ok(())
    }

    pub fn list_chunks(&self) -> Result<Vec<StoredChunk>> {
        let connection = self.connection.lock();
        let mut statement =
            connection.prepare("SELECT id, document_id, text FROM chunks ORDER BY id")?;
        let rows = statement.query_map([], chunk_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn chunk(&self, id: u64) -> Result<Option<StoredChunk>> {
        let connection = self.connection.lock();
        connection
            .query_row(
                "SELECT id, document_id, text FROM chunks WHERE id = ?",
                [id as i64],
                chunk_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn chunks_for_document(&self, document_id: &str) -> Result<Vec<StoredChunk>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, document_id, text FROM chunks WHERE document_id = ? ORDER BY position",
        )?;
        let rows = statement.query_map([document_id], chunk_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn chunks_for_document_limited(
        &self,
        document_id: &str,
        limit: usize,
    ) -> Result<Vec<StoredChunk>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, document_id, text FROM chunks
             WHERE document_id = ? ORDER BY position LIMIT ?",
        )?;
        let rows =
            statement.query_map(params![document_id, limit.max(1) as i64], chunk_from_row)?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn chunk_page(&self, offset: usize, limit: usize) -> Result<Vec<IndexedChunk>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT c.id, c.document_id, d.name, d.path, c.position, c.text
             FROM chunks c JOIN documents d ON d.id = c.document_id
             ORDER BY d.name COLLATE NOCASE, d.path, c.position LIMIT ? OFFSET ?",
        )?;
        let rows = statement.query_map(params![limit as i64, offset as i64], |row| {
            Ok(IndexedChunk {
                id: row.get::<_, i64>(0)? as u64,
                document_id: row.get(1)?,
                document_name: row.get(2)?,
                document_path: row.get(3)?,
                position: row.get::<_, i64>(4)? as usize,
                text: row.get(5)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    #[cfg(test)]
    pub fn failures(&self) -> Result<Vec<IndexFailure>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT path, category, reason FROM failures ORDER BY updated_at DESC, path",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(IndexFailure {
                path: row.get(0)?,
                category: row.get(1)?,
                reason: row.get(2)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn failure_page(&self, offset: usize, limit: usize) -> Result<Vec<IndexFailure>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT path, category, reason FROM failures
             ORDER BY updated_at DESC, path LIMIT ? OFFSET ?",
        )?;
        let rows = statement.query_map(params![limit as i64, offset as i64], |row| {
            Ok(IndexFailure {
                path: row.get(0)?,
                category: row.get(1)?,
                reason: row.get(2)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn file_timing_count(&self) -> Result<usize> {
        query_count(
            &self.connection.lock(),
            "SELECT COUNT(*) FROM file_index_timings",
        )
    }

    pub fn file_timing_page(&self, offset: usize, limit: usize) -> Result<Vec<FileIndexTiming>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT path, extension, size, status, check_ms, parse_ms, ocr_ms, embedding_ms,
                    storage_ms, text_index_ms, total_ms, recorded_at
             FROM file_index_timings ORDER BY total_ms DESC, recorded_at DESC LIMIT ? OFFSET ?",
        )?;
        let rows = statement.query_map(params![limit as i64, offset as i64], |row| {
            Ok(FileIndexTiming {
                path: row.get(0)?,
                extension: row.get(1)?,
                size: row.get::<_, i64>(2)?.max(0) as u64,
                status: row.get(3)?,
                check_ms: row.get::<_, i64>(4)?.max(0) as u64,
                parse_ms: row.get::<_, i64>(5)?.max(0) as u64,
                ocr_ms: row.get::<_, i64>(6)?.max(0) as u64,
                embedding_ms: row.get::<_, i64>(7)?.max(0) as u64,
                storage_ms: row.get::<_, i64>(8)?.max(0) as u64,
                text_index_ms: row.get::<_, i64>(9)?.max(0) as u64,
                total_ms: row.get::<_, i64>(10)?.max(0) as u64,
                recorded_at: row.get(11)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn record_file_timing(&self, timing: &FileIndexTiming) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute(
            "INSERT INTO file_index_timings(path, extension, size, status, check_ms, parse_ms, ocr_ms,
                    embedding_ms, storage_ms, text_index_ms, total_ms, recorded_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(path) DO UPDATE SET extension=excluded.extension, size=excluded.size,
               status=excluded.status, check_ms=excluded.check_ms, parse_ms=excluded.parse_ms, ocr_ms=excluded.ocr_ms,
               embedding_ms=excluded.embedding_ms, storage_ms=excluded.storage_ms,
               text_index_ms=excluded.text_index_ms, total_ms=excluded.total_ms,
               recorded_at=excluded.recorded_at",
            params![
                timing.path,
                timing.extension,
                timing.size as i64,
                timing.status,
                timing.check_ms as i64,
                timing.parse_ms as i64,
                timing.ocr_ms as i64,
                timing.embedding_ms as i64,
                timing.storage_ms as i64,
                timing.text_index_ms as i64,
                timing.total_ms as i64,
                timing.recorded_at,
            ],
        )?;
        connection.execute(
            "DELETE FROM file_index_timings WHERE path IN (
               SELECT path FROM file_index_timings ORDER BY recorded_at DESC LIMIT -1 OFFSET 10000
             )",
            [],
        )?;
        Ok(())
    }

    pub fn add_file_timing_stages(
        &self,
        path: &str,
        embedding_ms: u64,
        storage_ms: u64,
        text_index_ms: u64,
        total_ms: u64,
    ) -> Result<()> {
        self.connection.lock().execute(
            "UPDATE file_index_timings SET embedding_ms=embedding_ms+?,
               storage_ms=storage_ms+?, text_index_ms=text_index_ms+?, total_ms=total_ms+?,
               recorded_at=? WHERE path=?",
            params![
                embedding_ms as i64,
                storage_ms as i64,
                text_index_ms as i64,
                total_ms as i64,
                chrono::Utc::now().to_rfc3339(),
                path,
            ],
        )?;
        Ok(())
    }

    pub fn clear_failures(&self) -> Result<()> {
        self.connection.lock().execute("DELETE FROM failures", [])?;
        Ok(())
    }

    pub fn record_failure(&self, failure: &IndexFailure) -> Result<()> {
        self.connection.lock().execute(
            "INSERT INTO failures(path, category, reason, updated_at) VALUES (?, ?, ?, ?)
             ON CONFLICT(path) DO UPDATE SET category=excluded.category, reason=excluded.reason, updated_at=excluded.updated_at",
            params![failure.path, failure.category, failure.reason, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn missing_embedding_count(&self) -> Result<usize> {
        let connection = self.connection.lock();
        query_count(
            &connection,
            &format!(
                "SELECT COUNT(*) FROM documents WHERE embedding IS NULL OR length(embedding) != {}",
                crate::embedding::EMBEDDING_DIMENSION * std::mem::size_of::<f32>()
            ),
        )
    }

    pub fn embedding_profile_matches(&self) -> Result<bool> {
        Ok(self.setting("embedding_profile")?.as_deref()
            == Some(crate::embedding::EMBEDDING_PROFILE))
    }

    pub fn clear_embeddings(&self) -> Result<()> {
        self.connection.lock().execute_batch(
            "UPDATE documents SET embedding = NULL;
             DELETE FROM chunk_embeddings;
             DELETE FROM semantic_passages;
             DELETE FROM semantic_anchors;",
        )?;
        self.embeddings.write().clear();
        self.chunk_embeddings.write().clear();
        self.anchor_embeddings.write().clear();
        Ok(())
    }

    pub fn set_embedding_profile(&self) -> Result<()> {
        self.set_setting("embedding_profile", crate::embedding::EMBEDDING_PROFILE)
    }

    pub fn extraction_profile_matches(&self) -> Result<bool> {
        Ok(self.setting("extraction_profile")?.as_deref()
            == Some(crate::extract::EXTRACTION_PROFILE))
    }

    pub fn set_extraction_profile(&self) -> Result<()> {
        self.set_setting("extraction_profile", crate::extract::EXTRACTION_PROFILE)
    }

    fn setting(&self, key: &str) -> Result<Option<String>> {
        self.connection
            .lock()
            .query_row("SELECT value FROM settings WHERE key = ?", [key], |row| {
                row.get(0)
            })
            .optional()
            .map_err(Into::into)
    }

    fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.connection.lock().execute(
            "INSERT INTO settings(key, value) VALUES (?, ?)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }
}

fn load_embedding_cache(connection: &Connection) -> Result<HashMap<String, CachedEmbedding>> {
    let mut statement = connection
        .prepare("SELECT id, extension, embedding FROM documents WHERE embedding IS NOT NULL")?;
    let rows = statement.query_map([], |row| {
        let bytes: Vec<u8> = row.get(2)?;
        Ok((
            row.get::<_, String>(0)?,
            CachedEmbedding {
                extension: row.get(1)?,
                vector: blob_to_vector(&bytes),
            },
        ))
    })?;
    rows.collect::<rusqlite::Result<HashMap<_, _>>>()
        .map_err(Into::into)
}

fn load_chunk_embedding_cache(
    connection: &Connection,
    anchors: bool,
) -> Result<HashMap<u64, CachedChunkEmbedding>> {
    let table = if anchors {
        "semantic_anchors"
    } else {
        "semantic_passages"
    };
    let mut statement = connection.prepare(&format!(
        "SELECT ce.id, ce.document_id, d.extension, ce.embedding, ce.text, ce.chunk_id
         FROM {table} ce JOIN documents d ON d.id = ce.document_id"
    ))?;
    let rows = statement.query_map([], |row| {
        let bytes: Vec<u8> = row.get(3)?;
        Ok((
            row.get::<_, i64>(0)? as u64,
            CachedChunkEmbedding {
                chunk_id: row.get::<_, i64>(5)? as u64,
                document_id: row.get(1)?,
                extension: row.get(2)?,
                vector: blob_to_vector(&bytes),
                text: row.get(4)?,
            },
        ))
    })?;
    rows.collect::<rusqlite::Result<HashMap<_, _>>>()
        .map_err(Into::into)
}

fn query_count(connection: &Connection, sql: &str) -> Result<usize> {
    Ok(connection.query_row(sql, [], |row| row.get::<_, i64>(0))? as usize)
}

fn text_fingerprint(chunks: &[String]) -> String {
    let mut digest = Sha256::new();
    for chunk in chunks {
        digest.update((chunk.len() as u64).to_le_bytes());
        digest.update(chunk.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn document_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredDocument> {
    let embedding: Option<Vec<u8>> = row.get(7)?;
    Ok(StoredDocument {
        id: row.get(0)?,
        root: row.get(1)?,
        path: row.get(2)?,
        name: row.get(3)?,
        extension: row.get(4)?,
        modified_ms: row.get::<_, i64>(5)? as u64,
        size: row.get::<_, i64>(6)? as u64,
        embedding: embedding.map(|bytes| blob_to_vector(&bytes)),
    })
}

fn chunk_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredChunk> {
    Ok(StoredChunk {
        id: row.get::<_, i64>(0)? as u64,
        document_id: row.get(1)?,
        text: row.get(2)?,
    })
}

fn passage_cache_key(input: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(crate::embedding::EMBEDDING_PROFILE.as_bytes());
    digest.update([0]);
    digest.update(input.as_bytes());
    format!("{:x}", digest.finalize())
}

fn vector_to_blob(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn blob_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn sqlite_path(data_dir: &Path) -> PathBuf {
    data_dir.join("search.db")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{fallback_embed, EMBEDDING_DIMENSION};

    #[test]
    fn anchors_use_local_evidence_survive_restart_and_follow_document_lifecycle() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("search.db");
        let mut document = PreparedDocument {
            id: "structured".into(),
            root: "/test".into(),
            path: "/test/source.pdf".into(),
            name: "source.pdf".into(),
            extension: "pdf".into(),
            modified_ms: 1,
            size: 10,
            chunks: vec!["local body".into(), "other section".into()],
            embedding: Vec::new(),
        };
        {
            let storage = Storage::open(&path).unwrap();
            let chunks = storage.upsert_document(&document).unwrap();
            storage
                .replace_chunk_embeddings(&[
                    (chunks[0].id, "local body".into(), vec![0.0, 1.0]),
                    (chunks[1].id, "other section".into(), vec![1.0, 0.0]),
                ])
                .unwrap();
            storage
                .replace_anchor_embeddings(&[(chunks[0].id, "title".into(), vec![1.0, 0.0])])
                .unwrap();
        }
        let storage = Storage::open(&path).unwrap();
        let hits = storage.semantic_anchor_search(&[1.0, 0.0], Some("pdf"));
        assert_eq!(
            hits,
            vec![("structured".into(), "local body".into(), 1.0, 0.0)]
        );
        assert!(storage
            .semantic_anchor_search(&[1.0, 0.0], Some("xlsx"))
            .is_empty());
        assert_eq!(
            storage.semantic_chunk_search(&[1.0, 0.0], None, 10)[0].1,
            "other section"
        );
        document.chunks = vec!["updated".into()];
        let chunks = storage.upsert_document(&document).unwrap();
        assert!(storage.semantic_anchor_search(&[1.0, 0.0], None).is_empty());
        storage
            .replace_chunk_embeddings(&[(chunks[0].id, "updated".into(), vec![1.0, 0.0])])
            .unwrap();
        storage
            .replace_anchor_embeddings(&[(chunks[0].id, "new title".into(), vec![1.0, 0.0])])
            .unwrap();
        storage.clear_embeddings().unwrap();
        assert!(storage.anchor_embeddings.read().is_empty());
        storage
            .replace_anchor_embeddings(&[(chunks[0].id, "new title".into(), vec![1.0, 0.0])])
            .unwrap();
        storage.delete_document(&document.id).unwrap();
        drop(storage);
        assert!(Storage::open(&path)
            .unwrap()
            .anchor_embeddings
            .read()
            .is_empty());
    }

    #[test]
    fn vectors_round_trip_as_compact_blobs() {
        let vector = vec![0.25, -0.5, 1.0];
        assert_eq!(blob_to_vector(&vector_to_blob(&vector)), vector);
    }

    #[test]
    fn passage_cache_survives_restart_and_requires_identical_input_and_profile() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.db");
        let input = "novel.txt\nchapter one".to_owned();
        let vector = fallback_embed("chapter one");
        {
            let storage = Storage::open(&path).unwrap();
            storage
                .cache_passage_vectors(std::slice::from_ref(&input), std::slice::from_ref(&vector))
                .unwrap();
            storage
                .connection
                .lock()
                .execute(
                    "INSERT INTO passage_vector_cache VALUES ('old-profile-key', ?, 0)",
                    [vector_to_blob(&vector)],
                )
                .unwrap();
        }
        let storage = Storage::open(&path).unwrap();
        let inputs = vec![
            input,
            "novel.txt\nchapter two".to_owned(),
            "other.txt\nchapter one".to_owned(),
        ];
        let cached = storage.cached_passage_vectors(&inputs).unwrap();
        assert_eq!(cached, vec![Some(vector), None, None]);
    }

    #[test]
    fn passage_cache_rejects_and_replaces_invalid_vectors() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("cache.db")).unwrap();
        let inputs = vec!["novel.txt\nchapter one".to_owned()];
        for invalid in [vec![1.0], vec![f32::NAN; EMBEDDING_DIMENSION]] {
            storage.cache_passage_vectors(&inputs, &[invalid]).unwrap();
            assert_eq!(storage.cached_passage_vectors(&inputs).unwrap(), vec![None]);
        }
        let vector = fallback_embed("chapter one");
        storage
            .cache_passage_vectors(&inputs, std::slice::from_ref(&vector))
            .unwrap();
        assert_eq!(
            storage.cached_passage_vectors(&inputs).unwrap(),
            vec![Some(vector)]
        );
    }

    #[test]
    fn previous_quantization_profile_requires_reindex_and_cannot_reuse_vectors() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.db");
        let input = "novel.txt\nchapter one".to_owned();
        let previous = "rwkv7-tiny-mixed-int8-structured-full-passages-v9";
        let vector = fallback_embed("chapter one");
        {
            let storage = Storage::open(&path).unwrap();
            storage.set_setting("embedding_profile", previous).unwrap();
            let mut digest = Sha256::new();
            digest.update(previous.as_bytes());
            digest.update([0]);
            digest.update(input.as_bytes());
            storage
                .connection
                .lock()
                .execute(
                    "INSERT INTO passage_vector_cache VALUES (?, ?, 0)",
                    params![format!("{:x}", digest.finalize()), vector_to_blob(&vector)],
                )
                .unwrap();
            assert!(!storage.embedding_profile_matches().unwrap());
            assert_eq!(
                storage
                    .cached_passage_vectors(std::slice::from_ref(&input))
                    .unwrap(),
                vec![None]
            );
            storage
                .cache_passage_vectors(std::slice::from_ref(&input), std::slice::from_ref(&vector))
                .unwrap();
            storage.set_embedding_profile().unwrap();
        }
        let storage = Storage::open(&path).unwrap();
        assert!(storage.embedding_profile_matches().unwrap());
        assert_eq!(
            storage.cached_passage_vectors(&[input]).unwrap(),
            vec![Some(vector)]
        );
    }

    #[test]
    fn semantic_windows_survive_restart_and_follow_document_lifecycle() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("search.db");
        let storage = Storage::open(&path).unwrap();
        let mut document = PreparedDocument {
            id: "first".to_owned(),
            root: "/test".to_owned(),
            path: "/test/first.txt".to_owned(),
            name: "first.txt".to_owned(),
            extension: "txt".to_owned(),
            modified_ms: 1,
            size: 12,
            chunks: vec!["first window and second window".to_owned()],
            embedding: Vec::new(),
        };
        let chunks = storage.upsert_document(&document).unwrap();
        let fingerprint = storage.content_fingerprints()["first"].clone();
        storage
            .replace_chunk_embeddings(&[
                (chunks[0].id, "first window".to_owned(), vec![1.0, 0.0]),
                (chunks[0].id, "second window".to_owned(), vec![0.0, 1.0]),
            ])
            .unwrap();
        assert!(!storage.has_embedding("first"));
        document.embedding = vec![1.0, 0.0];
        storage
            .update_document_embeddings(std::slice::from_ref(&document))
            .unwrap();
        assert_eq!(
            storage.chunk(chunks[0].id).unwrap().unwrap().text,
            document.chunks[0]
        );
        drop(storage);

        let storage = Storage::open(&path).unwrap();
        assert!(storage.has_embedding("first"));
        assert_eq!(storage.content_fingerprints()["first"], fingerprint);
        assert_eq!(storage.chunk_embeddings.read().len(), 2);
        assert_eq!(
            storage.semantic_chunk_search(&[0.0, 1.0], None, 10)[0].1,
            "second window"
        );
        assert_eq!(
            storage.semantic_chunk_search(&[1.0, 0.0], None, 10).len(),
            1
        );
        assert!(storage
            .semantic_chunk_search(&[1.0, 0.0], Some("pdf"), 10)
            .is_empty());
        storage
            .replace_chunk_embeddings(&[(chunks[0].id, "first window".to_owned(), vec![0.0, 1.0])])
            .unwrap();
        assert_eq!(storage.chunk_embeddings.read().len(), 2);

        document.chunks = vec!["updated body".to_owned()];
        document.embedding.clear();
        let chunks = storage.upsert_document(&document).unwrap();
        assert_ne!(storage.content_fingerprints()["first"], fingerprint);
        assert!(!storage.has_embedding("first"));
        assert!(storage
            .semantic_chunk_search(&[1.0, 0.0], None, 10)
            .is_empty());
        storage
            .replace_chunk_embeddings(&[(chunks[0].id, "updated body".to_owned(), vec![1.0, 0.0])])
            .unwrap();
        storage.delete_document("first").unwrap();
        assert!(storage.content_fingerprints().is_empty());
        assert!(storage.chunk_embeddings.read().is_empty());
        drop(storage);
        let storage = Storage::open(&path).unwrap();
        assert!(storage.chunk_embeddings.read().is_empty());
        assert!(storage.content_fingerprints().is_empty());
    }

    #[test]
    fn semantic_candidates_include_distinct_documents_before_limit() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        for (id, vector) in [("first", vec![1.0, 0.0]), ("second", vec![0.8, 0.6])] {
            let chunks = storage
                .upsert_document(&PreparedDocument {
                    id: id.to_owned(),
                    root: "/test".to_owned(),
                    path: format!("/test/{id}.txt"),
                    name: format!("{id}.txt"),
                    extension: "txt".to_owned(),
                    modified_ms: 1,
                    size: 12,
                    chunks: vec!["first window and second window".to_owned()],
                    embedding: Vec::new(),
                })
                .unwrap();
            storage
                .replace_chunk_embeddings(&[
                    (chunks[0].id, "first window".to_owned(), vector.clone()),
                    (chunks[0].id, "second window".to_owned(), vector),
                ])
                .unwrap();
        }
        let hits = storage.semantic_chunk_search(&[1.0, 0.0], None, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "first");
        assert_eq!(hits[1].0, "second");
        storage.clear_embeddings().unwrap();
        assert!(storage
            .semantic_chunk_search(&[1.0, 0.0], None, 2)
            .is_empty());
        assert_eq!(storage.content_fingerprints().len(), 2);
    }

    #[test]
    fn semantic_confirmation_uses_the_same_passage_as_the_original_query() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let chunks = storage
            .upsert_document(&PreparedDocument {
                id: "deep".into(),
                root: "/test".into(),
                path: "/test/deep.txt".into(),
                name: "deep.txt".into(),
                extension: "txt".into(),
                modified_ms: 1,
                size: 100,
                chunks: vec!["first body".into(), "deep body".into()],
                embedding: Vec::new(),
            })
            .unwrap();
        storage
            .replace_chunk_embeddings(&[
                (chunks[0].id, "first body".into(), vec![0.0, 1.0]),
                (chunks[1].id, "deep body".into(), vec![1.0, 0.0]),
            ])
            .unwrap();
        let hits = storage.semantic_chunk_matches(&[1.0, 0.0], &[0.0, 1.0], None, 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "deep body");
        assert_eq!(hits[0].score, 1.0);
        assert_eq!(hits[0].confirmation, 0.0);
        assert!(storage
            .semantic_chunk_matches(&[1.0, 0.0], &[0.0, 1.0], Some("pdf"), 10)
            .is_empty());
    }

    #[test]
    fn semantic_windows_preserve_keyword_candidates_and_document_limits() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let mut preferred = Vec::new();
        for identity in ["first", "second"] {
            let chunks = storage
                .upsert_document(&PreparedDocument {
                    id: identity.into(),
                    root: "/test".into(),
                    path: format!("/test/{identity}.txt"),
                    name: format!("{identity}.txt"),
                    extension: "txt".into(),
                    modified_ms: 1,
                    size: 100,
                    chunks: vec!["top".into(), "specific".into(), "other".into()],
                    embedding: Vec::new(),
                })
                .unwrap();
            storage
                .replace_chunk_embeddings(&[
                    (chunks[0].id, "top".into(), vec![1.0, 0.0]),
                    (chunks[1].id, "specific".into(), vec![0.6, 0.8]),
                    (chunks[2].id, "other".into(), vec![0.8, 0.6]),
                ])
                .unwrap();
            preferred.push(chunks[1].id);
        }
        let hits =
            storage.semantic_passage_matches(&[1.0, 0.0], &[0.0, 1.0], None, 2, 1, &preferred);
        assert_eq!(hits.len(), 4);
        for identity in ["first", "second"] {
            let specific = hits
                .iter()
                .find(|hit| hit.document_id == identity && hit.text == "specific")
                .unwrap();
            assert_eq!(specific.score, 0.6);
            assert_eq!(specific.confirmation, 0.8);
            assert!(preferred.contains(&specific.chunk_id));
        }
        let limited =
            storage.semantic_passage_matches(&[1.0, 0.0], &[0.0, 1.0], None, 1, 2, &preferred);
        assert_eq!(limited.len(), 3);
        assert!(limited
            .iter()
            .all(|hit| hit.document_id == limited[0].document_id));
        assert!(storage
            .semantic_passage_matches(&[1.0, 0.0], &[0.0, 1.0], Some("pdf"), 2, 2, &preferred)
            .is_empty());
    }

    #[test]
    fn semantic_windows_keep_joint_evidence_outside_the_raw_shortlist() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let chunks = storage
            .upsert_document(&PreparedDocument {
                id: "document".into(),
                root: "/test".into(),
                path: "/test/document.txt".into(),
                name: "document.txt".into(),
                extension: "txt".into(),
                modified_ms: 1,
                size: 100,
                chunks: vec!["original high and jointly supported deep passage".into()],
                embedding: Vec::new(),
            })
            .unwrap();
        let mut passages = (0..30)
            .map(|position| {
                (
                    chunks[0].id,
                    format!("raw-{position}"),
                    vec![0.99 - position as f32 * 0.001, 0.01],
                )
            })
            .collect::<Vec<_>>();
        passages.push((chunks[0].id, "joint evidence".into(), vec![0.60, 0.80]));
        storage.replace_chunk_embeddings(&passages).unwrap();
        let hits = storage.semantic_passage_matches(&[1.0, 0.0], &[0.0, 1.0], None, 1, 2, &[]);
        assert!(hits.len() <= 4);
        assert!(hits.iter().any(|hit| hit.text == "raw-0"));
        let joint = hits
            .iter()
            .find(|hit| hit.text == "joint evidence")
            .unwrap();
        assert_eq!(joint.score, 0.60);
        assert_eq!(joint.confirmation, 0.80);
        assert!(storage
            .semantic_passage_matches(&[1.0, 0.0], &[0.0, 1.0], None, 0, 2, &[])
            .is_empty());
    }

    #[test]
    fn inventory_pages_return_documents_chunks_and_failures() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        storage
            .upsert_document(&PreparedDocument {
                id: "doc-1".to_owned(),
                root: "/inventory".to_owned(),
                path: "/inventory/示例.txt".to_owned(),
                name: "示例.txt".to_owned(),
                extension: "txt".to_owned(),
                modified_ms: 1,
                size: 12,
                chunks: vec!["第一段".to_owned(), "第二段".to_owned()],
                embedding: fallback_embed("示例"),
            })
            .unwrap();
        storage
            .record_failure(&IndexFailure {
                path: "/inventory/损坏.pdf".to_owned(),
                category: "corrupt".to_owned(),
                reason: "解析失败".to_owned(),
            })
            .unwrap();

        assert_eq!(storage.document_page(0, 50).unwrap().len(), 1);
        assert_eq!(storage.chunk_page(0, 50).unwrap().len(), 2);
        assert_eq!(storage.failure_page(0, 50).unwrap().len(), 1);
    }

    #[test]
    fn file_timing_records_are_sorted_and_stage_updates_accumulate() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        for (path, total_ms) in [("/slow.pdf", 900), ("/quick.txt", 100)] {
            storage
                .record_file_timing(&FileIndexTiming {
                    path: path.to_owned(),
                    extension: path.rsplit('.').next().unwrap().to_owned(),
                    size: 1024,
                    status: "indexed".to_owned(),
                    check_ms: 5,
                    parse_ms: total_ms,
                    ocr_ms: 0,
                    embedding_ms: 0,
                    storage_ms: 0,
                    text_index_ms: 0,
                    total_ms,
                    recorded_at: chrono::Utc::now().to_rfc3339(),
                })
                .unwrap();
        }
        storage
            .add_file_timing_stages("/slow.pdf", 40, 20, 10, 70)
            .unwrap();

        let page = storage.file_timing_page(0, 1).unwrap();
        assert_eq!(storage.file_timing_count().unwrap(), 2);
        assert_eq!(page[0].path, "/slow.pdf");
        assert_eq!(page[0].embedding_ms, 40);
        assert_eq!(page[0].storage_ms, 20);
        assert_eq!(page[0].text_index_ms, 10);
        assert_eq!(page[0].total_ms, 970);
        assert_eq!(
            storage.file_timing_page(1, 1).unwrap()[0].path,
            "/quick.txt"
        );
    }

    #[test]
    #[ignore = "显式运行的 50,000 文件容量测试"]
    fn sqlite_wal_handles_fifty_thousand_documents() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        for index in 0..50_000 {
            let path = format!("/capacity/{index}.txt");
            storage
                .upsert_document(&PreparedDocument {
                    id: format!("doc-{index}"),
                    root: "/capacity".to_owned(),
                    path: path.clone(),
                    name: format!("{index}.txt"),
                    extension: "txt".to_owned(),
                    modified_ms: index,
                    size: 32,
                    chunks: vec![format!("第 {index} 份容量测试文档")],
                    embedding: fallback_embed(&path),
                })
                .unwrap();
        }
        let counts = storage.counts().unwrap();
        assert_eq!(counts.documents, 50_000);
        assert_eq!(counts.chunks, 50_000);
    }
}
