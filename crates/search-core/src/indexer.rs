use crate::embedding::{normalize, EmbeddingEngine, EMBEDDING_DIMENSION};
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
#[cfg(not(test))]
use std::process::{Command, Stdio};
use std::sync::OnceLock;
#[cfg(not(test))]
use std::thread::sleep;
use std::time::{Duration, Instant};
#[cfg(not(test))]
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

const MAX_FILE_BYTES: u64 = 200 * 1024 * 1024;
const EMBEDDING_BATCH_SIZE: usize = 4;
const PARSE_BATCH_SIZE: usize = 16;
const CHUNK_TARGET: usize = 800;
const CHUNK_OVERLAP: usize = 100;
const MAX_SEMANTIC_CHUNKS_PER_DOCUMENT: usize = 16;
const MAX_SEMANTIC_EMBED_CHARS: usize = 480;
const SEMANTIC_WINDOW_STRIDE: usize = 400;
#[cfg(not(test))]
const FILE_EXTRACTION_TIMEOUT: Duration = Duration::from_secs(180);

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
    let mut documents = std::mem::take(pending);
    let semantic_chunks = documents
        .iter()
        .map(semantic_chunks_for_document)
        .collect::<Vec<_>>();
    report_progress(progress, processed, total, None, "storage", accumulated);
    let started = Instant::now();
    let chunks = storage.upsert_documents(&documents)?;
    timings.storage += started.elapsed();
    report_progress_with(
        progress,
        processed,
        total,
        None,
        "text_index",
        accumulated.scan,
        accumulated.check,
        accumulated.parse,
        accumulated.embedding,
        accumulated.storage + timings.storage,
        accumulated.text_index,
    );
    let started = Instant::now();
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
    text_index.replace_documents(&updates)?;
    text_index.commit()?;
    timings.text_index += started.elapsed();

    let embedding_tasks = semantic_embedding_tasks(&semantic_chunks);
    if !embedding_tasks.is_empty() {
        let current_file = documents.first().map(|document| document.path.as_str());
        report_progress_with(
            progress,
            processed,
            total,
            current_file,
            "embedding",
            accumulated.scan,
            accumulated.check,
            accumulated.parse,
            accumulated.embedding + timings.embedding,
            accumulated.storage + timings.storage,
            accumulated.text_index + timings.text_index,
        );
        let inputs = embedding_tasks
            .iter()
            .map(|(document_index, _, text)| {
                semantic_chunk_text(&documents[*document_index].name, text)
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        let embeddings = embedder.embed_passages(&inputs);
        timings.embedding += started.elapsed();
        anyhow::ensure!(
            embeddings.len() == embedding_tasks.len(),
            "Missing semantic passage vectors"
        );
        let mut vectors_by_document = vec![Vec::new(); documents.len()];
        let updates = embedding_tasks
            .into_iter()
            .zip(embeddings)
            .map(|((document_index, position, text), embedding)| {
                vectors_by_document[document_index].push((position, embedding.clone()));
                (chunks[document_index][position].id, text, embedding)
            })
            .collect::<Vec<_>>();
        report_progress_with(
            progress,
            processed,
            total,
            current_file,
            "storage",
            accumulated.scan,
            accumulated.check,
            accumulated.parse,
            accumulated.embedding + timings.embedding,
            accumulated.storage + timings.storage,
            accumulated.text_index + timings.text_index,
        );
        let started = Instant::now();
        storage.replace_chunk_embeddings(&updates)?;
        timings.storage += started.elapsed();
        for (document, vectors) in documents.iter_mut().zip(vectors_by_document) {
            document.embedding = average_embedding(&vectors);
        }
    }
    report_progress_with(
        progress,
        processed,
        total,
        None,
        "storage",
        accumulated.scan,
        accumulated.check,
        accumulated.parse,
        accumulated.embedding + timings.embedding,
        accumulated.storage + timings.storage,
        accumulated.text_index + timings.text_index,
    );
    let started = Instant::now();
    storage.update_document_embeddings(&documents)?;
    timings.storage += started.elapsed();
    Ok(timings)
}

fn semantic_chunks_for_document(document: &PreparedDocument) -> Vec<(usize, String)> {
    let lengths = document
        .chunks
        .iter()
        .map(|text| text.chars().count())
        .collect::<Vec<_>>();
    let counts = lengths
        .iter()
        .map(|length| {
            if *length == 0 {
                0
            } else if *length <= MAX_SEMANTIC_EMBED_CHARS {
                1
            } else {
                (length - MAX_SEMANTIC_EMBED_CHARS).div_ceil(SEMANTIC_WINDOW_STRIDE) + 1
            }
        })
        .collect::<Vec<_>>();
    let window_count = counts.iter().sum::<usize>();
    let sample_count = window_count.min(MAX_SEMANTIC_CHUNKS_PER_DOCUMENT);
    let selected = (0..sample_count)
        .map(|index| {
            if sample_count <= 1 {
                0
            } else {
                index * (window_count - 1) / (sample_count - 1)
            }
        })
        .collect::<HashSet<_>>();
    let mut offset = 0;
    let mut seen = HashSet::new();
    let mut passages = Vec::new();
    for (position, count) in counts.into_iter().enumerate() {
        let selected_windows = (0..count)
            .filter(|index| selected.contains(&(offset + index)))
            .collect::<Vec<_>>();
        if !selected_windows.is_empty() {
            let chars = document.chunks[position].chars().collect::<Vec<_>>();
            for index in selected_windows {
                let start = index * SEMANTIC_WINDOW_STRIDE;
                let end = (start + MAX_SEMANTIC_EMBED_CHARS).min(chars.len());
                let text = chars[start..end].iter().collect::<String>();
                if seen.insert(text.clone()) {
                    passages.push((position, text));
                }
            }
        }
        offset += count;
    }
    passages
}

fn semantic_chunk_text(name: &str, text: &str) -> String {
    let name = name.chars().take(160).collect::<String>();
    format!("{name}\n{text}")
}

fn semantic_embedding_tasks(
    semantic_chunks: &[Vec<(usize, String)>],
) -> Vec<(usize, usize, String)> {
    semantic_chunks
        .iter()
        .enumerate()
        .flat_map(|(document_index, passages)| {
            passages
                .iter()
                .map(move |(position, text)| (document_index, *position, text.clone()))
        })
        .collect()
}

fn average_embedding(vectors: &[(usize, Vec<f32>)]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let mut average = vec![0.0; EMBEDDING_DIMENSION];
    let mut count = 0;
    for (_, vector) in vectors {
        if vector.len() != EMBEDDING_DIMENSION {
            continue;
        }
        for (target, value) in average.iter_mut().zip(vector) {
            *target += value;
        }
        count += 1;
    }
    if count == 0 {
        return Vec::new();
    }
    let divisor = count as f32;
    for value in &mut average {
        *value /= divisor;
    }
    normalize(&mut average);
    average
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
                let path = job.path;
                let started = Instant::now();
                let result = prepare_document(&path, &job.root, job.modified_ms, job.size);
                (path, started.elapsed(), result)
            })
            .collect::<Vec<_>>()
    });
    timings.parse += started.elapsed();
    let mut prepared = false;
    for (job_index, (path, elapsed, result)) in results.into_iter().enumerate() {
        if elapsed >= Duration::from_secs(2) {
            tracing::info!(
                path = %path.display(),
                elapsed_ms = elapsed.as_millis(),
                "slow document extraction"
            );
        }
        match result {
            Ok(document) => {
                pending.push(document);
                prepared = true;
            }
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    elapsed_ms = elapsed.as_millis(),
                    error = %format_args!("{error:#}"),
                    "document extraction failed"
                );
                storage.record_failure(&failure_for(&path, &error))?;
            }
        }
        if pending.len() >= EMBEDDING_BATCH_SIZE {
            let mut batch = pending.drain(..EMBEDDING_BATCH_SIZE).collect::<Vec<_>>();
            let persisted = persist_batch(
                &mut batch,
                storage,
                text_index,
                embedder,
                progress,
                processed + job_index + 1,
                total,
                timings,
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
    let started = Instant::now();
    let text = extract_text_with_timeout(path)?;
    tracing::debug!(
        path = %path.display(),
        elapsed_ms = started.elapsed().as_millis(),
        "document extracted"
    );
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

fn extract_text_with_timeout(path: &Path) -> Result<String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "pdf" | "docx" | "xlsx" | "pptx") {
        return extract_text(path);
    }

    #[cfg(test)]
    {
        return extract_text(path);
    }

    #[cfg(not(test))]
    {
        extract_text_in_worker(path)
    }
}

#[cfg(not(test))]
fn extract_text_in_worker(path: &Path) -> Result<String> {
    let executable = std::env::current_exe().context("无法定位搜索服务程序")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let output = std::env::temp_dir().join(format!(
        "filesearch-extract-{}-{}-{}.txt",
        std::process::id(),
        nonce,
        stable_id(&path.to_string_lossy())
    ));
    let mut child = Command::new(executable)
        .arg("--extract-file")
        .arg(path)
        .arg("--extract-output")
        .arg(&output)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("无法启动文件解析 worker: {}", path.display()))?;
    let deadline = Instant::now() + FILE_EXTRACTION_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                let _ = std::fs::remove_file(&output);
                anyhow::bail!("文档解析 worker 失败，退出码: {status}");
            }
            let text = std::fs::read_to_string(&output)
                .with_context(|| format!("无法读取文件解析结果: {}", path.display()))?;
            let _ = std::fs::remove_file(&output);
            return Ok(text);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&output);
            anyhow::bail!(
                "文档解析超时（{} 秒），已跳过文件: {}",
                FILE_EXTRACTION_TIMEOUT.as_secs(),
                path.display()
            );
        }
        sleep(Duration::from_millis(100));
    }
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
    } else if lower.contains("超时") || lower.contains("timeout") {
        "timeout"
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
    fn semantic_chunks_cover_document_positions_with_a_bound() {
        let document = PreparedDocument {
            id: "id".to_owned(),
            root: "root".to_owned(),
            name: "长文档.docx".to_owned(),
            extension: "docx".to_owned(),
            path: "长文档.docx".to_owned(),
            modified_ms: 0,
            size: 0,
            chunks: (0..20).map(|index| format!("片段 {index}")).collect(),
            embedding: Vec::new(),
        };
        let samples = semantic_chunks_for_document(&document);
        assert_eq!(samples.len(), MAX_SEMANTIC_CHUNKS_PER_DOCUMENT);
        assert_eq!(samples.first().map(|sample| sample.0), Some(0));
        assert_eq!(samples.last().map(|sample| sample.0), Some(19));
    }

    #[test]
    fn semantic_chunk_text_preserves_contiguous_text() {
        let text = format!("开头{}结尾", "中".repeat(400));
        let value = semantic_chunk_text("测试.txt", &text);
        assert!(value.chars().count() <= "测试.txt".chars().count() + MAX_SEMANTIC_EMBED_CHARS + 2);
        assert!(value.starts_with("测试.txt\n开头"));
        assert!(value.ends_with("结尾"));
    }

    #[test]
    fn semantic_embedding_tasks_preserve_document_and_window_mapping() {
        let passages = vec![
            vec![(2, "first tail".to_owned()), (0, "first head".to_owned())],
            vec![(1, "second middle".to_owned())],
        ];
        assert_eq!(
            semantic_embedding_tasks(&passages),
            vec![
                (0, 2, "first tail".to_owned()),
                (0, 0, "first head".to_owned()),
                (1, 1, "second middle".to_owned()),
            ]
        );
    }

    #[test]
    fn semantic_windows_cover_middle_and_end_without_joining_distant_text() {
        let text = format!("{}中心答案{}结尾", "前".repeat(500), "后".repeat(500));
        let document = PreparedDocument {
            id: "id".to_owned(),
            root: "root".to_owned(),
            name: "test.txt".to_owned(),
            extension: "txt".to_owned(),
            path: "test.txt".to_owned(),
            modified_ms: 0,
            size: 0,
            chunks: vec![text.clone()],
            embedding: Vec::new(),
        };
        let passages = semantic_chunks_for_document(&document);
        assert_eq!(passages.len(), 3);
        assert!(passages.iter().any(|(_, text)| text.contains("中心答案")));
        assert!(passages.last().unwrap().1.ends_with("结尾"));
        assert!(passages
            .iter()
            .all(|(_, passage)| text.contains(passage) && passage.chars().count() <= 480));
    }

    #[test]
    fn average_embedding_combines_and_normalizes_chunk_vectors() {
        let first = vec![1.0; EMBEDDING_DIMENSION];
        let second = vec![-1.0; EMBEDDING_DIMENSION];
        let average = average_embedding(&[(0, first.clone()), (1, first)]);
        assert!((average.iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 0.0001);
        assert!(average.iter().all(|value| *value > 0.0));
        assert!(average_embedding(&[(0, second)])
            .iter()
            .all(|value| *value < 0.0));
    }

    #[test]
    fn failure_categories_cover_required_cases() {
        assert_eq!(classify_failure("document is encrypted"), "encrypted");
        assert_eq!(classify_failure("Permission denied"), "permission");
        assert_eq!(classify_failure("network share offline"), "offline");
        assert_eq!(classify_failure("invalid zip archive"), "corrupt");
        assert_eq!(classify_failure("文档解析超时（180 秒）"), "timeout");
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
