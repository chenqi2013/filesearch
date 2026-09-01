use crate::embedding::{EmbeddingEngine, EMBEDDING_DIMENSION};
use crate::extract::{extract_text, is_supported};
use crate::model::{IndexFailure, PreparedDocument};
use crate::storage::Storage;
use crate::text_index::TextIndex;
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use walkdir::WalkDir;

const MAX_FILE_BYTES: u64 = 200 * 1024 * 1024;
const EMBEDDING_BATCH_SIZE: usize = 32;
const PARSE_BATCH_SIZE: usize = 16;
const CHUNK_TARGET: usize = 800;
const CHUNK_OVERLAP: usize = 100;

#[derive(Default)]
struct IndexTimings {
    scan: Duration,
    check: Duration,
    parse: Duration,
    embedding: Duration,
    storage: Duration,
    text_index: Duration,
}

struct ParseJob {
    path: PathBuf,
    root: String,
    modified_ms: u64,
    size: u64,
}

#[derive(Default)]
struct PersistTimings {
    embedding: Duration,
    storage: Duration,
    text_index: Duration,
}

pub struct ProgressUpdate<'a> {
    pub processed: usize,
    pub total: usize,
    pub current_file: Option<&'a str>,
    pub stage: &'static str,
    pub scan_ms: u128,
    pub check_ms: u128,
    pub parse_ms: u128,
    pub embedding_ms: u128,
    pub storage_ms: u128,
    pub text_index_ms: u128,
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
    let started = Instant::now();
    let mut timings = IndexTimings::default();
    let roots = normalize_roots(paths);
    storage.set_directories(&roots)?;
    storage.clear_failures()?;

    let previous = storage.list_documents_without_embeddings()?;
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

    let scan_started = Instant::now();
    report_progress(&mut progress, 0, 0, None, "scanning", &timings);
    let mut scans = Vec::with_capacity(roots.len());
    for root in &roots {
        scans.push(scan_root(root, storage));
    }
    let scan_elapsed = scan_started.elapsed();
    timings.scan = scan_elapsed;
    let total = scans.iter().map(|scan| scan.files.len()).sum::<usize>();
    report_progress(&mut progress, 0, total, None, "checking", &timings);

    let mut processed = 0usize;
    let mut pending = Vec::with_capacity(EMBEDDING_BATCH_SIZE);
    let mut parse_jobs = Vec::with_capacity(PARSE_BATCH_SIZE);
    let mut seen_by_root: HashMap<String, HashSet<String>> = HashMap::new();

    for scan in &scans {
        let seen = seen_by_root.entry(scan.path.clone()).or_default();
        for path in &scan.files {
            let path_string = path.to_string_lossy().into_owned();
            report_progress(
                &mut progress,
                processed,
                total,
                Some(&path_string),
                "checking",
                &timings,
            );
            seen.insert(path_string.clone());
            let check_started = Instant::now();
            match metadata(path) {
                Ok((modified_ms, size)) => {
                    if previous_by_path
                        .get(path_string.as_str())
                        .is_some_and(|existing| {
                            existing.modified_ms == modified_ms
                                && existing.size == size
                                && storage.has_embedding(&existing.id)
                        })
                    {
                        processed += 1;
                        timings.check += check_started.elapsed();
                        continue;
                    }
                    parse_jobs.push(ParseJob {
                        path: path.clone(),
                        root: scan.path.clone(),
                        modified_ms,
                        size,
                    });
                }
                Err(error) => {
                    storage.record_failure(&failure_for(path, &error))?;
                    processed += 1;
                }
            }
            timings.check += check_started.elapsed();
            if parse_jobs.len() >= PARSE_BATCH_SIZE {
                let (completed, prepared) = flush_parse_jobs(
                    &mut parse_jobs,
                    &mut pending,
                    storage,
                    text_index,
                    embedder,
                    &mut timings,
                    &mut progress,
                    processed,
                    total,
                )?;
                processed += completed;
                changed |= prepared;
                report_progress(&mut progress, processed, total, None, "checking", &timings);
            }
        }
    }
    if !parse_jobs.is_empty() {
        let (completed, prepared) = flush_parse_jobs(
            &mut parse_jobs,
            &mut pending,
            storage,
            text_index,
            embedder,
            &mut timings,
            &mut progress,
            processed,
            total,
        )?;
        processed += completed;
        changed |= prepared;
    }
    if !pending.is_empty() {
        let persisted = persist_batch(
            &mut pending,
            storage,
            text_index,
            embedder,
            &mut progress,
            processed,
            total,
            &timings,
        )?;
        add_persist_timings(&mut timings, persisted);
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
        report_progress(
            &mut progress,
            processed,
            total,
            None,
            "committing",
            &timings,
        );
        let commit_started = Instant::now();
        text_index.commit()?;
        timings.text_index += commit_started.elapsed();
    }
    storage.set_embedding_profile()?;
    storage.set_last_indexed_now()?;
    report_progress(&mut progress, total, total, None, "ready", &timings);
    tracing::info!(
        files = total,
        processed,
        scan_ms = scan_elapsed.as_millis(),
        parse_ms = timings.parse.as_millis(),
        embedding_ms = timings.embedding.as_millis(),
        storage_ms = timings.storage.as_millis(),
        text_index_ms = timings.text_index.as_millis(),
        total_ms = started.elapsed().as_millis(),
        "completed full index"
    );
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
    let started = Instant::now();
    let mut timings = IndexTimings::default();
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
    report_progress(&mut progress, 0, files.len(), None, "checking", &timings);
    let mut pending = Vec::with_capacity(EMBEDDING_BATCH_SIZE);
    let mut parse_jobs = Vec::with_capacity(PARSE_BATCH_SIZE);
    let mut processed = 0usize;
    for (index, path) in files.iter().enumerate() {
        let path_string = path.to_string_lossy().into_owned();
        report_progress(
            &mut progress,
            index,
            files.len(),
            Some(&path_string),
            "checking",
            &timings,
        );
        let root = configured_roots
            .iter()
            .filter(|root| path.starts_with(root))
            .max_by_key(|root| root.len())
            .cloned();
        let Some(root) = root else {
            processed += 1;
            continue;
        };
        let check_started = Instant::now();
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
            Ok(Some(ParseJob {
                path: path.clone(),
                root,
                modified_ms,
                size,
            }))
        }) {
            Ok(Some(job)) => parse_jobs.push(job),
            Ok(None) => processed += 1,
            Err(error) => {
                storage.record_failure(&failure_for(path, &error))?;
                processed += 1;
            }
        }
        timings.check += check_started.elapsed();
        if parse_jobs.len() >= PARSE_BATCH_SIZE {
            let (completed, _) = flush_parse_jobs(
                &mut parse_jobs,
                &mut pending,
                storage,
                text_index,
                embedder,
                &mut timings,
                &mut progress,
                processed,
                files.len(),
            )?;
            processed += completed;
        }
    }
    if !parse_jobs.is_empty() {
        let (completed, _) = flush_parse_jobs(
            &mut parse_jobs,
            &mut pending,
            storage,
            text_index,
            embedder,
            &mut timings,
            &mut progress,
            processed,
            files.len(),
        )?;
        processed += completed;
    }
    if !pending.is_empty() {
        let persisted = persist_batch(
            &mut pending,
            storage,
            text_index,
            embedder,
            &mut progress,
            processed,
            files.len(),
            &timings,
        )?;
        add_persist_timings(&mut timings, persisted);
    }
    report_progress(
        &mut progress,
        processed,
        files.len(),
        None,
        "committing",
        &timings,
    );
    let commit_started = Instant::now();
    text_index.commit()?;
    timings.text_index += commit_started.elapsed();
    storage.set_last_indexed_now()?;
    report_progress(
        &mut progress,
        files.len(),
        files.len(),
        None,
        "ready",
        &timings,
    );
    tracing::info!(
        files = files.len(),
        processed,
        parse_ms = timings.parse.as_millis(),
        embedding_ms = timings.embedding.as_millis(),
        storage_ms = timings.storage.as_millis(),
        text_index_ms = timings.text_index.as_millis(),
        total_ms = started.elapsed().as_millis(),
        "completed incremental index"
    );
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
    progress: &mut dyn FnMut(ProgressUpdate<'_>),
    processed: usize,
    total: usize,
    accumulated: &IndexTimings,
) -> Result<PersistTimings> {
    let mut timings = PersistTimings::default();
    let inputs = pending.iter().map(semantic_source).collect::<Vec<_>>();
    let embedding_file = pending
        .iter()
        .zip(&inputs)
        .max_by_key(|(_, input)| input.chars().count())
        .map(|(document, _)| document.path.as_str());
    report_progress(
        progress,
        processed,
        total,
        embedding_file,
        "embedding",
        accumulated,
    );
    let started = Instant::now();
    let embeddings = embedder.embed_passages(&inputs);
    timings.embedding = started.elapsed();
    let documents = pending
        .drain(..)
        .zip(embeddings)
        .map(|(mut document, embedding)| {
            document.embedding = embedding;
            document
        })
        .collect::<Vec<_>>();
    let embedding_total = accumulated.embedding + timings.embedding;
    report_progress_with(
        progress,
        processed,
        total,
        None,
        "storage",
        accumulated.scan,
        accumulated.check,
        accumulated.parse,
        embedding_total,
        accumulated.storage,
        accumulated.text_index,
    );
    let started = Instant::now();
    let chunks = storage.upsert_documents(&documents)?;
    timings.storage = started.elapsed();
    let updates = documents
        .iter()
        .zip(&chunks)
        .map(|(document, chunks)| {
            (
                document.id.as_str(),
                document.name.as_str(),
                chunks.as_slice(),
            )
        })
        .collect::<Vec<_>>();
    let storage_total = accumulated.storage + timings.storage;
    report_progress_with(
        progress,
        processed,
        total,
        None,
        "text_index",
        accumulated.scan,
        accumulated.check,
        accumulated.parse,
        embedding_total,
        storage_total,
        accumulated.text_index,
    );
    let started = Instant::now();
    text_index.replace_documents(&updates)?;
    timings.text_index = started.elapsed();
    Ok(timings)
}

fn semantic_source(document: &PreparedDocument) -> String {
    const MAX_SEMANTIC_CHARS: usize = 2_000;
    const PREFIX_CHARS: usize = 800;
    const SAMPLE_COUNT: usize = 4;
    let mut source = format!("{}\n", document.name);
    let body = document_text_chars(&document.chunks);
    let body_budget = MAX_SEMANTIC_CHARS.saturating_sub(source.chars().count());
    if body.len() <= body_budget {
        source.extend(body);
        source.push('\n');
        return source;
    }

    let separator_budget = SAMPLE_COUNT + 1;
    let content_budget = body_budget.saturating_sub(separator_budget);
    let prefix_length = PREFIX_CHARS.min(content_budget).min(body.len());
    source.extend(&body[..prefix_length]);
    let mut remaining = content_budget.saturating_sub(prefix_length);
    let sample_length = remaining.div_ceil(SAMPLE_COUNT).min(300);
    let span = body.len().saturating_sub(prefix_length + sample_length);
    for sample_index in 0..SAMPLE_COUNT {
        if remaining == 0 {
            break;
        }
        let length = sample_length.min(remaining);
        let start = prefix_length + span.saturating_mul(sample_index + 1) / SAMPLE_COUNT;
        let end = (start + length).min(body.len());
        source.push('\n');
        source.extend(&body[start..end]);
        remaining = remaining.saturating_sub(end - start);
    }
    source.push('\n');
    source
}

fn document_text_chars(chunks: &[String]) -> Vec<char> {
    let mut output = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let chars = chunk.chars().collect::<Vec<_>>();
        let skip = if index == 0 {
            0
        } else {
            CHUNK_OVERLAP.min(chars.len())
        };
        output.extend_from_slice(&chars[skip..]);
    }
    output
}

fn flush_parse_jobs(
    jobs: &mut Vec<ParseJob>,
    pending: &mut Vec<PreparedDocument>,
    storage: &Storage,
    text_index: &TextIndex,
    embedder: &EmbeddingEngine,
    timings: &mut IndexTimings,
    progress: &mut dyn FnMut(ProgressUpdate<'_>),
    processed: usize,
    total: usize,
) -> Result<(usize, bool)> {
    let jobs = std::mem::take(jobs);
    let completed = jobs.len();
    let current_file = jobs
        .first()
        .map(|job| job.path.to_string_lossy().into_owned());
    report_progress(
        progress,
        processed,
        total,
        current_file.as_deref(),
        "parsing",
        timings,
    );
    let started = Instant::now();
    let results = parse_pool().install(|| {
        jobs.into_par_iter()
            .map(|job| {
                let result = prepare_document(&job.path, &job.root, job.modified_ms, job.size);
                (job.path, result)
            })
            .collect::<Vec<_>>()
    });
    timings.parse += started.elapsed();
    let mut prepared = false;
    for (path, result) in results {
        match result {
            Ok(document) => {
                pending.push(document);
                prepared = true;
            }
            Err(error) => storage.record_failure(&failure_for(&path, &error))?,
        }
        if pending.len() >= EMBEDDING_BATCH_SIZE {
            let mut batch = pending.drain(..EMBEDDING_BATCH_SIZE).collect::<Vec<_>>();
            let persisted = persist_batch(
                &mut batch, storage, text_index, embedder, progress, processed, total, timings,
            )?;
            add_persist_timings(timings, persisted);
        }
    }
    Ok((completed, prepared))
}

fn report_progress(
    progress: &mut dyn FnMut(ProgressUpdate<'_>),
    processed: usize,
    total: usize,
    current_file: Option<&str>,
    stage: &'static str,
    timings: &IndexTimings,
) {
    report_progress_with(
        progress,
        processed,
        total,
        current_file,
        stage,
        timings.scan,
        timings.check,
        timings.parse,
        timings.embedding,
        timings.storage,
        timings.text_index,
    );
}

#[allow(clippy::too_many_arguments)]
fn report_progress_with(
    progress: &mut dyn FnMut(ProgressUpdate<'_>),
    processed: usize,
    total: usize,
    current_file: Option<&str>,
    stage: &'static str,
    scan: Duration,
    check: Duration,
    parse: Duration,
    embedding: Duration,
    storage: Duration,
    text_index: Duration,
) {
    progress(ProgressUpdate {
        processed,
        total,
        current_file,
        stage,
        scan_ms: scan.as_millis(),
        check_ms: check.as_millis(),
        parse_ms: parse.as_millis(),
        embedding_ms: embedding.as_millis(),
        storage_ms: storage.as_millis(),
        text_index_ms: text_index.as_millis(),
    });
}

fn parse_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let thread_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(1, 4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(|index| format!("filesearch-parser-{index}"))
            .build()
            .expect("无法创建文档解析线程池")
    })
}

fn add_persist_timings(timings: &mut IndexTimings, persisted: PersistTimings) {
    timings.embedding += persisted.embedding;
    timings.storage += persisted.storage;
    timings.text_index += persisted.text_index;
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
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= CHUNK_TARGET {
        return vec![text.to_owned()];
    }
    let mut output = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + CHUNK_TARGET).min(chars.len());
        output.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start = end - CHUNK_OVERLAP;
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
    fn semantic_source_samples_the_end_of_long_documents() {
        let document = PreparedDocument {
            id: "id".to_owned(),
            root: "root".to_owned(),
            name: "长文档.docx".to_owned(),
            extension: "docx".to_owned(),
            path: "长文档.docx".to_owned(),
            modified_ms: 0,
            size: 0,
            chunks: split_chunks(&format!("{}结尾关键内容", "前".repeat(6_000))),
            embedding: Vec::new(),
        };
        let source = semantic_source(&document);
        assert!(source.contains("结尾关键内容"));
        assert!(source.chars().count() <= 2_000);
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
