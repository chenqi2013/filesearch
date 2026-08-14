use crate::extract::{extract_text, is_supported};
use crate::model::{ChunkRecord, DocumentRecord, IndexFailure, PersistedIndex};
use crate::search::{chunks, embed, sorted_tokens};
use anyhow::Result;
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct ProgressUpdate<'a> {
    pub processed: usize,
    pub total: usize,
    pub current_file: Option<&'a str>,
}

pub fn build_index<F>(
    paths: &[String],
    previous: &PersistedIndex,
    mut progress: F,
) -> PersistedIndex
where
    F: FnMut(ProgressUpdate<'_>),
{
    let roots = paths
        .iter()
        .map(PathBuf::from)
        .filter(|path| path.exists() && path.is_dir())
        .collect::<Vec<_>>();
    let files = roots
        .iter()
        .flat_map(|root| {
            WalkDir::new(root)
                .follow_links(false)
                .into_iter()
                .filter_map(Result::ok)
        })
        .filter(|entry| entry.file_type().is_file() && is_supported(entry.path()))
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();
    progress(ProgressUpdate {
        processed: 0,
        total: files.len(),
        current_file: None,
    });

    let previous_by_path = previous
        .documents
        .iter()
        .map(|document| (document.path.as_str(), document))
        .collect::<HashMap<_, _>>();
    let mut documents = Vec::new();
    let mut failures = Vec::new();
    let mut seen = HashSet::new();

    for (index, path) in files.iter().enumerate() {
        let path_string = path.to_string_lossy().into_owned();
        progress(ProgressUpdate {
            processed: index,
            total: files.len(),
            current_file: Some(&path_string),
        });
        seen.insert(path_string.clone());
        match metadata(path) {
            Ok((modified_ms, size)) => {
                if let Some(existing) = previous_by_path.get(path_string.as_str()) {
                    if existing.modified_ms == modified_ms && existing.size == size {
                        documents.push((*existing).clone());
                        continue;
                    }
                }
                match extract_document(path, modified_ms, size) {
                    Ok(document) => documents.push(document),
                    Err(error) => failures.push(IndexFailure {
                        path: path_string,
                        reason: format!("{error:#}"),
                    }),
                }
            }
            Err(error) => failures.push(IndexFailure {
                path: path_string,
                reason: format!("{error:#}"),
            }),
        }
    }

    progress(ProgressUpdate {
        processed: files.len(),
        total: files.len(),
        current_file: None,
    });
    PersistedIndex {
        directories: roots
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        documents,
        failures,
        last_indexed: Some(Utc::now().to_rfc3339()),
    }
}

pub fn load(path: &Path) -> PersistedIndex {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub fn save(path: &Path, index: &PersistedIndex) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec(index)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn metadata(path: &Path) -> Result<(u64, u64)> {
    let metadata = fs::metadata(path)?;
    let modified_ms = metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    Ok((modified_ms, metadata.len()))
}

fn extract_document(path: &Path, modified_ms: u64, size: u64) -> Result<DocumentRecord> {
    if size > 100 * 1024 * 1024 {
        anyhow::bail!("文件超过 MVP 的 100 MB 单文件限制");
    }
    let text = extract_text(path)?;
    let chunk_records = chunks(&text)
        .into_iter()
        .map(|text| ChunkRecord {
            embedding: embed(&text),
            terms: sorted_tokens(&text),
            text,
        })
        .collect::<Vec<_>>();
    let path_string = path.to_string_lossy().into_owned();
    Ok(DocumentRecord {
        id: stable_id(&path_string),
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
        chunks: chunk_records,
    })
}

fn stable_id(value: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
