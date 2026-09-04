use crate::model::{
    IndexFailure, IndexedChunk, IndexedDocument, LegacyIndex, PreparedDocument, StoredChunk,
    StoredDocument,
};
use anyhow::{Context, Result};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Storage {
    connection: Mutex<Connection>,
    embeddings: RwLock<HashMap<String, CachedEmbedding>>,
    chunk_embeddings: RwLock<HashMap<u64, CachedChunkEmbedding>>,
}

struct CachedEmbedding {
    extension: String,
    vector: Vec<f32>,
}

struct CachedChunkEmbedding {
    document_id: String,
    extension: String,
    vector: Vec<f32>,
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
             CREATE TABLE IF NOT EXISTS failures (
               path TEXT PRIMARY KEY,
               category TEXT NOT NULL,
               reason TEXT NOT NULL,
               updated_at TEXT NOT NULL
             );",
        )?;
        let embeddings = load_embedding_cache(&connection)?;
        let chunk_embeddings = load_chunk_embedding_cache(&connection)?;
        Ok(Self {
            connection: Mutex::new(connection),
            embeddings: RwLock::new(embeddings),
            chunk_embeddings: RwLock::new(chunk_embeddings),
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
        let mut cache = self.embeddings.write();
        for document in documents {
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
        self.chunk_embeddings
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

    pub fn semantic_chunk_search(
        &self,
        query: &[f32],
        extension: Option<&str>,
        limit: usize,
    ) -> Vec<(String, u64, f32)> {
        let cache = self.chunk_embeddings.read();
        let mut scores = cache
            .par_iter()
            .filter(|(_, embedding)| extension.is_none_or(|value| value == embedding.extension))
            .map(|(chunk_id, embedding)| {
                (
                    embedding.document_id.clone(),
                    *chunk_id,
                    crate::embedding::cosine(query, &embedding.vector).max(0.0),
                )
            })
            .collect::<Vec<_>>();
        scores.sort_unstable_by(|left, right| right.2.total_cmp(&left.2));
        scores.truncate(limit);
        scores
    }

    pub fn replace_chunk_embeddings(&self, embeddings: &[(u64, Vec<f32>)]) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }
        let mut connection = self.connection.lock();
        let transaction = connection.transaction()?;
        let mut cached = Vec::with_capacity(embeddings.len());
        for (chunk_id, embedding) in embeddings {
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
            transaction.execute(
                "INSERT OR REPLACE INTO chunk_embeddings(chunk_id, document_id, embedding)
                 VALUES (?, ?, ?)",
                params![*chunk_id as i64, &document_id, vector_to_blob(embedding)],
            )?;
            cached.push((
                *chunk_id,
                CachedChunkEmbedding {
                    document_id,
                    extension,
                    vector: embedding.clone(),
                },
            ));
        }
        transaction.commit()?;
        drop(connection);
        let mut cache = self.chunk_embeddings.write();
        for (chunk_id, embedding) in cached {
            cache.insert(chunk_id, embedding);
        }
        Ok(())
    }

    pub fn has_embedding(&self, document_id: &str) -> bool {
        self.embeddings.read().contains_key(document_id)
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
             DELETE FROM chunk_embeddings;",
        )?;
        self.embeddings.write().clear();
        self.chunk_embeddings.write().clear();
        Ok(())
    }

    pub fn set_embedding_profile(&self) -> Result<()> {
        self.set_setting("embedding_profile", crate::embedding::EMBEDDING_PROFILE)
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
) -> Result<HashMap<u64, CachedChunkEmbedding>> {
    let mut statement = connection.prepare(
        "SELECT ce.chunk_id, ce.document_id, d.extension, ce.embedding
         FROM chunk_embeddings ce JOIN documents d ON d.id = ce.document_id",
    )?;
    let rows = statement.query_map([], |row| {
        let bytes: Vec<u8> = row.get(3)?;
        Ok((
            row.get::<_, i64>(0)? as u64,
            CachedChunkEmbedding {
                document_id: row.get(1)?,
                extension: row.get(2)?,
                vector: blob_to_vector(&bytes),
            },
        ))
    })?;
    rows.collect::<rusqlite::Result<HashMap<_, _>>>()
        .map_err(Into::into)
}

fn query_count(connection: &Connection, sql: &str) -> Result<usize> {
    Ok(connection.query_row(sql, [], |row| row.get::<_, i64>(0))? as usize)
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
    use crate::embedding::fallback_embed;

    #[test]
    fn vectors_round_trip_as_compact_blobs() {
        let vector = vec![0.25, -0.5, 1.0];
        assert_eq!(blob_to_vector(&vector_to_blob(&vector)), vector);
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
