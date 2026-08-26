use crate::embedding::{cosine, lexical_terms, EmbeddingEngine};
use crate::model::{SearchMode, SearchRequest, SearchResult, StoredDocument};
use crate::storage::Storage;
use crate::text_index::TextIndex;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct Candidate {
    keyword: f32,
    semantic: f32,
    snippet: Option<String>,
}

pub fn search(
    storage: &Storage,
    text_index: &TextIndex,
    embedder: &EmbeddingEngine,
    request: &SearchRequest,
) -> Result<Vec<SearchResult>> {
    let extension = request
        .extension
        .as_deref()
        .filter(|value| !value.is_empty());
    let stored_documents = if request.mode == SearchMode::Keyword {
        // Keyword-only searches do not need roughly 146 MB of vectors at the
        // 50,000-document target size.
        storage.list_documents_without_embeddings()?
    } else {
        storage.list_documents()?
    };
    let documents = stored_documents
        .into_iter()
        .filter(|document| extension.is_none_or(|value| value == document.extension))
        .collect::<Vec<_>>();
    let by_id = documents
        .iter()
        .map(|document| (document.id.as_str(), document))
        .collect::<HashMap<_, _>>();
    let mut candidates: HashMap<String, Candidate> = HashMap::new();
    let candidate_limit = (request.limit.clamp(1, 100) * 20).clamp(200, 1_000);

    if request.mode != SearchMode::Semantic {
        let hits = text_index.search(&request.query, candidate_limit)?;
        let max_score = hits
            .first()
            .map(|hit| hit.score)
            .unwrap_or(1.0)
            .max(f32::EPSILON);
        for hit in hits {
            let Some(chunk) = storage.chunk(hit.chunk_id)? else {
                continue;
            };
            if !by_id.contains_key(chunk.document_id.as_str()) {
                continue;
            }
            let normalized = (hit.score / max_score).clamp(0.0, 1.0);
            let candidate = candidates.entry(chunk.document_id).or_default();
            if normalized > candidate.keyword {
                candidate.keyword = normalized;
                candidate.snippet = Some(chunk.text);
            }
        }
    }

    if request.mode != SearchMode::Keyword {
        let query_embedding = embedder.embed_query(&request.query);
        let mut semantic = documents
            .iter()
            .filter_map(|document| {
                let embedding = document.embedding.as_deref()?;
                let score = cosine(&query_embedding, embedding).max(0.0);
                (score > 0.05).then(|| (document.id.clone(), score))
            })
            .collect::<Vec<_>>();
        semantic.sort_by(|left, right| right.1.total_cmp(&left.1));
        semantic.truncate(candidate_limit);
        for (document_id, score) in semantic {
            candidates.entry(document_id).or_default().semantic = score;
        }
    }

    let query_lower = request.query.to_lowercase();
    let mut ranked = candidates
        .into_iter()
        .filter_map(|(document_id, candidate)| {
            let document = by_id.get(document_id.as_str())?;
            let filename_boost = if document.name.to_lowercase().contains(&query_lower) {
                0.12
            } else {
                0.0
            };
            let score = match request.mode {
                SearchMode::Keyword => candidate.keyword,
                SearchMode::Semantic => candidate.semantic,
                SearchMode::Hybrid => candidate.keyword * 0.62 + candidate.semantic * 0.38,
            } + filename_boost;
            let threshold = match request.mode {
                SearchMode::Keyword => 0.01,
                SearchMode::Semantic => 0.12,
                SearchMode::Hybrid => 0.06,
            };
            (score >= threshold).then_some((document_id, score, candidate.snippet))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    ranked.truncate(request.limit.clamp(1, 100));

    let mut results = Vec::with_capacity(ranked.len());
    for (document_id, score, matched_text) in ranked {
        let document = by_id[document_id.as_str()];
        let text = match matched_text {
            Some(text) => text,
            None => best_semantic_snippet(storage, document, &request.query)?,
        };
        results.push(SearchResult {
            id: document.id.clone(),
            path: document.path.clone(),
            name: document.name.clone(),
            extension: document.extension.clone(),
            modified_ms: document.modified_ms,
            size: document.size,
            snippet: snippet(&text, &request.query),
            score: ((score.clamp(0.0, 1.0) * 100.0).round()) / 100.0,
        });
    }
    Ok(results)
}

fn best_semantic_snippet(
    storage: &Storage,
    document: &StoredDocument,
    query: &str,
) -> Result<String> {
    let query_terms = lexical_terms(query).into_iter().collect::<HashSet<_>>();
    let mut chunks = storage.chunks_for_document(&document.id)?;
    chunks.sort_by_key(|chunk| {
        let terms = lexical_terms(&chunk.text);
        std::cmp::Reverse(
            terms
                .iter()
                .filter(|term| query_terms.contains(*term))
                .count(),
        )
    });
    Ok(chunks
        .into_iter()
        .next()
        .map(|chunk| chunk.text)
        .unwrap_or_default())
}

fn snippet(text: &str, query: &str) -> String {
    const WINDOW: usize = 260;
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= WINDOW {
        return text.to_owned();
    }
    let lower = text.to_lowercase();
    let query_lower = query.to_lowercase();
    let byte_position = lower.find(&query_lower).unwrap_or(0);
    let char_position = lower[..byte_position].chars().count();
    let start = char_position.saturating_sub(70);
    let end = (start + WINDOW).min(chars.len());
    let mut value = chars[start..end].iter().collect::<String>();
    if start > 0 {
        value.insert_str(0, "...");
    }
    if end < chars.len() {
        value.push_str("...");
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_keeps_result_bounded() {
        let text = format!("{}目标词{}", "前".repeat(400), "后".repeat(400));
        let value = snippet(&text, "目标词");
        assert!(value.chars().count() <= 266);
        assert!(value.contains("目标词"));
    }
}
