use crate::embedding::{EmbeddingEngine, EMBEDDING_DIMENSION};
use crate::extract::{extract_text, is_supported};
use crate::model::{IndexFailure, PreparedDocument};
use crate::storage::Storage;
use crate::text_index::TextIndex;
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const MAX_FILE_BYTES: u64 = 200 * 1024 * 1024;
const EMBEDDING_BATCH_SIZE: usize = 32;

pub struct ProgressUpdate<'a> {
    pub processed: usize,
    pub total: usize,
    pub current_file: Option<&'a str>,
}

struct ScannedRoot {
    path: String,
    complete: bool,
    files: Vec<PathBuf>,
}

pub fn build_index<F>(
    paths: &[String],
    storage: &Storage,
    text_index: &TextIndex,
    embedder: &EmbeddingEngine,
    mut progress: F,
) -> Result<()>
where
    F: FnMut(ProgressUpdate<'_>),
{
    let roots = normalize_roots(paths);
    storage.set_directories(&roots)?;
    storage.clear_failures()?;

    let previous = storage.list_documents()?;
    let previous_by_path = previous
        .iter()
        .map(|document| (document.path.as_str(), document))
        .collect::<HashMap<_, _>>();
    let configured = roots.iter().cloned().collect::<HashSet<_>>();
    let mut changed = false;

    for document in previous
        .iter()
        .filter(|document| !configured.contains(&document.root))
    {
        text_index.delete_document(&document.id);
        storage.delete_document(&document.id)?;
        changed = true;
    }

    let mut scans = Vec::with_capacity(roots.len());
    for root in &roots {
        scans.push(scan_root(root, storage));
    }
    let total = scans.iter().map(|scan| scan.files.len()).sum::<usize>();
    progress(ProgressUpdate {
        processed: 0,
        total,
        current_file: None,
    });

    let mut processed = 0usize;
    let mut pending = Vec::with_capacity(EMBEDDING_BATCH_SIZE);
    let mut seen_by_root: HashMap<String, HashSet<String>> = HashMap::new();

    for scan in &scans {
        let seen = seen_by_root.entry(scan.path.clone()).or_default();
        for path in &scan.files {
            let path_string = path.to_string_lossy().into_owned();
            progress(ProgressUpdate {
                processed,
                total,
                current_file: Some(&path_string),
            });
            seen.insert(path_string.clone());
            match metadata(path) {
                Ok((modified_ms, size)) => {
                    if previous_by_path
                        .get(path_string.as_str())
                        .is_some_and(|existing| {
                            existing.modified_ms == modified_ms
                                && existing.size == size
                                && existing
                                    .embedding
                                    .as_ref()
                                    .is_some_and(|embedding| embedding.len() == EMBEDDING_DIMENSION)
                        })
                    {
                        processed += 1;
                        continue;
                    }
                    match prepare_document(path, &scan.path, modified_ms, size) {
                        Ok(document) => pending.push(document),
                        Err(error) => {
                            storage.record_failure(&failure_for(path, &error))?;
                        }
                    }
                }
                Err(error) => storage.record_failure(&failure_for(path, &error))?,
            }
            processed += 1;
            if pending.len() >= EMBEDDING_BATCH_SIZE {
                persist_batch(&mut pending, storage, text_index, embedder)?;
                changed = true;
            }
        }
    }
    if !pending.is_empty() {
        persist_batch(&mut pending, storage, text_index, embedder)?;
        changed = true;
    }

    for scan in &scans {
        if !scan.complete {
            continue;
        }
        let seen = seen_by_root.get(&scan.path).cloned().unwrap_or_default();
        for document in previous
            .iter()
            .filter(|document| document.root == scan.path && !seen.contains(&document.path))
        {
            text_index.delete_document(&document.id);
            storage.delete_document(&document.id)?;
            changed = true;
        }
    }

    if changed {
        text_index.commit()?;
    }
    storage.set_embedding_profile()?;
    storage.set_last_indexed_now()?;
    progress(ProgressUpdate {
        processed: total,
        total,
        current_file: None,
    });
    Ok(())
}

pub fn update_paths<F>(
    paths: &[PathBuf],
    storage: &Storage,
    text_index: &TextIndex,
    embedder: &EmbeddingEngine,
    mut progress: F,
) -> Result<()>
where
    F: FnMut(ProgressUpdate<'_>),
{
    let configured_roots = storage.directories()?;
    let mut candidates = HashSet::new();
    // Watcher updates are frequent. Keep this lookup lightweight instead of
    // loading every document embedding for each event batch.
    let document_paths = storage.list_document_paths()?;

    for path in paths {
        if path.is_dir() {
            for entry in WalkDir::new(path)
                .follow_links(false)
                .into_iter()
                .filter_map(Result::ok)
            {
                if entry.file_type().is_file() && is_supported(entry.path()) {
                    candidates.insert(entry.into_path());
                }
            }
        } else if path.exists() {
            if is_supported(path) {
                candidates.insert(path.clone());
            }
        } else {
            let prefix = path.to_string_lossy();
            for (document_id, _document_path) in
                document_paths.iter().filter(|(_, document_path)| {
                    document_path.as_str() == prefix || Path::new(document_path).starts_with(path)
                })
            {
                text_index.delete_document(document_id);
                storage.delete_document(document_id)?;
            }
        }
    }

    let files = candidates.into_iter().collect::<Vec<_>>();
    progress(ProgressUpdate {
        processed: 0,
        total: files.len(),
        current_file: None,
    });
    let mut pending = Vec::with_capacity(EMBEDDING_BATCH_SIZE);
    for (index, path) in files.iter().enumerate() {
        let path_string = path.to_string_lossy().into_owned();
        progress(ProgressUpdate {
            processed: index,
            total: files.len(),
            current_file: Some(&path_string),
        });
        let root = configured_roots
            .iter()
            .filter(|root| path.starts_with(root))
            .max_by_key(|root| root.len())
            .cloned();
        let Some(root) = root else { continue };
        match metadata(path).and_then(|(modified_ms, size)| {
            if storage
                .document_by_path(&path_string)?
                .is_some_and(|existing| {
                    existing.modified_ms == modified_ms
                        && existing.size == size
                        && existing
                            .embedding
                            .as_ref()
                            .is_some_and(|embedding| embedding.len() == EMBEDDING_DIMENSION)
                })
            {
                return Ok(None);
            }
            prepare_document(path, &root, modified_ms, size).map(Some)
        }) {
            Ok(Some(document)) => pending.push(document),
            Ok(None) => {}
            Err(error) => storage.record_failure(&failure_for(path, &error))?,
        }
    }
    if !pending.is_empty() {
        persist_batch(&mut pending, storage, text_index, embedder)?;
    }
    text_index.commit()?;
    storage.set_last_indexed_now()?;
    progress(ProgressUpdate {
        processed: files.len(),
        total: files.len(),
        current_file: None,
    });
    Ok(())
}

fn scan_root(root: &str, storage: &Storage) -> ScannedRoot {
    let path = PathBuf::from(root);
    if !path.is_dir() {
        let _ = storage.record_failure(&IndexFailure {
            path: root.to_owned(),
            category: "offline".to_owned(),
            reason: "索引目录不可访问，可能已断线、被移动或权限不足；保留已有索引".to_owned(),
        });
        return ScannedRoot {
            path: root.to_owned(),
            complete: false,
            files: Vec::new(),
        };
    }
    let mut complete = true;
    let mut files = Vec::new();
    for entry in WalkDir::new(&path).follow_links(false) {
        match entry {
            Ok(entry) if entry.file_type().is_file() && is_supported(entry.path()) => {
                files.push(entry.into_path());
            }
            Ok(_) => {}
            Err(error) => {
                complete = false;
                let error_path = error.path().unwrap_or(&path).to_path_buf();
                let failure = failure_for(&error_path, &anyhow::anyhow!(error.to_string()));
                let _ = storage.record_failure(&failure);
            }
        }
    }
    ScannedRoot {
        path: root.to_owned(),
        complete,
        files,
    }
}

fn normalize_roots(paths: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    paths
        .iter()
        .map(|path| PathBuf::from(path).to_string_lossy().into_owned())
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

fn persist_batch(
    pending: &mut Vec<PreparedDocument>,
    storage: &Storage,
    text_index: &TextIndex,
    embedder: &EmbeddingEngine,
) -> Result<()> {
    let inputs = pending.iter().map(semantic_source).collect::<Vec<_>>();
    let embeddings = embedder.embed_passages(&inputs);
    for (mut document, embedding) in pending.drain(..).zip(embeddings) {
        document.embedding = embedding;
        let chunks = storage.upsert_document(&document)?;
        text_index.replace_document(&document.id, &document.name, &chunks)?;
        storage.remove_failure(&document.path)?;
    }
    Ok(())
}

fn semantic_source(document: &PreparedDocument) -> String {
    const MAX_SEMANTIC_CHARS: usize = 4_000;
    const MAX_CHUNKS_PER_DOCUMENT: usize = 64;
    let mut source = format!("{}\n", document.name);
    for chunk in document.chunks.iter().take(MAX_CHUNKS_PER_DOCUMENT) {
        let remaining = MAX_SEMANTIC_CHARS.saturating_sub(source.chars().count());
        if remaining == 0 {
            break;
        }
        source.extend(chunk.chars().take(remaining));
        source.push('\n');
    }
    source
}

fn metadata(path: &Path) -> Result<(u64, u64)> {
    let metadata =
        fs::metadata(path).with_context(|| format!("无法读取元数据: {}", path.display()))?;
    let modified_ms = metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    Ok((modified_ms, metadata.len()))
}

fn prepare_document(
    path: &Path,
    root: &str,
    modified_ms: u64,
    size: u64,
) -> Result<PreparedDocument> {
    if size > MAX_FILE_BYTES {
        anyhow::bail!("文件超过 200 MB 单文件限制");
    }
    let text = extract_text(path)?;
    let path_string = path.to_string_lossy().into_owned();
    Ok(PreparedDocument {
        id: stable_id(&path_string),
        root: root.to_owned(),
        name: path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_owned(),
        extension: path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase(),
        path: path_string,
        modified_ms,
        size,
        chunks: split_chunks(&text),
        embedding: Vec::new(),
    })
}

pub fn split_chunks(text: &str) -> Vec<String> {
    const TARGET: usize = 800;
    const OVERLAP: usize = 100;
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= TARGET {
        return vec![text.to_owned()];
    }
    let mut output = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + TARGET).min(chars.len());
        output.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start = end - OVERLAP;
    }
    output
}

fn stable_id(value: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn failure_for(path: &Path, error: &anyhow::Error) -> IndexFailure {
    let reason = format!("{error:#}");
    IndexFailure {
        path: path.to_string_lossy().into_owned(),
        category: classify_failure(&reason).to_owned(),
        reason,
    }
}

pub(crate) fn classify_failure(reason: &str) -> &'static str {
    let lower = reason.to_lowercase();
    if lower.contains("encrypt") || lower.contains("password") || lower.contains("加密") {
        "encrypted"
    } else if lower.contains("permission denied")
        || lower.contains("access is denied")
        || lower.contains("权限")
    {
        "permission"
    } else if lower.contains("no such file")
        || lower.contains("not found")
        || lower.contains("断线")
        || lower.contains("offline")
    {
        "offline"
    } else if lower.contains("invalid")
        || lower.contains("corrupt")
        || lower.contains("无效")
        || lower.contains("解析失败")
    {
        "corrupt"
    } else {
        "parse"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::fallback_embed;

    #[test]
    fn chunks_keep_overlap() {
        let output = split_chunks(&"本".repeat(1700));
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].chars().count(), 800);
    }

    #[test]
    fn semantic_source_includes_document_name() {
        let document = PreparedDocument {
            id: "id".to_owned(),
            root: "root".to_owned(),
            name: "流程.docx".to_owned(),
            extension: "docx".to_owned(),
            path: "流程.docx".to_owned(),
            modified_ms: 0,
            size: 0,
            chunks: vec!["第一段".to_owned()],
            embedding: Vec::new(),
        };
        assert_eq!(semantic_source(&document), "流程.docx\n第一段\n");
    }

    #[test]
    fn failure_categories_cover_required_cases() {
        assert_eq!(classify_failure("document is encrypted"), "encrypted");
        assert_eq!(classify_failure("Permission denied"), "permission");
        assert_eq!(classify_failure("network share offline"), "offline");
        assert_eq!(classify_failure("invalid zip archive"), "corrupt");
    }

    #[test]
    fn disconnected_root_preserves_existing_index() {
        let directory = tempfile::tempdir().unwrap();
        let missing_root = directory.path().join("disconnected-share");
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let text_index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let document_path = missing_root
            .join("cached.txt")
            .to_string_lossy()
            .into_owned();
        let document = PreparedDocument {
            id: stable_id(&document_path),
            root: missing_root.to_string_lossy().into_owned(),
            path: document_path,
            name: "cached.txt".to_owned(),
            extension: "txt".to_owned(),
            modified_ms: 1,
            size: 6,
            chunks: vec!["已缓存的共享盘文档".to_owned()],
            embedding: fallback_embed("已缓存的共享盘文档"),
        };
        let chunks = storage.upsert_document(&document).unwrap();
        text_index
            .replace_document(&document.id, &document.name, &chunks)
            .unwrap();
        text_index.commit().unwrap();
        let embedder = EmbeddingEngine::new(directory.path().join("models"));

        build_index(
            &[missing_root.to_string_lossy().into_owned()],
            &storage,
            &text_index,
            &embedder,
            |_| {},
        )
        .unwrap();

        assert_eq!(storage.counts().unwrap().documents, 1);
        assert!(storage
            .failures()
            .unwrap()
            .iter()
            .any(|failure| failure.category == "offline"));
    }
}
