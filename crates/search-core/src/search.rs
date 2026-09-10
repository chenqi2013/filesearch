use crate::embedding::{query_terms, EmbeddingEngine};
use crate::model::{SearchMode, SearchRequest, SearchResult, StoredDocument};
use crate::storage::Storage;
use crate::text_index::TextIndex;
use anyhow::Result;
use std::collections::HashMap;

#[derive(Default)]
struct Candidate {
    keyword: f32,
    semantic: f32,
    snippet: Option<String>,
    semantic_snippet: Option<String>,
}

const SEMANTIC_SNIPPET_SCAN_LIMIT: usize = 1;
const RAW_SEMANTIC_MIN_SCORE: f32 = 0.24;
const SEMANTIC_RESULT_MIN_SCORE: f32 = 0.36;
const KEYWORD_MIN_COVERAGE: f32 = 0.25;
const SEMANTIC_ONLY_MIN_SCORE: f32 = 0.50;

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
    let stored_documents = storage.list_documents_without_embeddings()?;
    let documents = stored_documents
        .into_iter()
        .filter(|document| extension.is_none_or(|value| value == document.extension))
        .collect::<Vec<_>>();
    let by_id = documents
        .iter()
        .map(|document| (document.id.as_str(), document))
        .collect::<HashMap<_, _>>();
    let mut candidates: HashMap<String, Candidate> = HashMap::new();
    let mut keyword_snippets = HashMap::new();
    let terms = query_terms(&request.query);
    let candidate_limit = (request.limit.clamp(1, 100) * 20).clamp(200, 1_000);

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
        let Some(document) = by_id.get(chunk.document_id.as_str()) else {
            continue;
        };
        let coverage = term_coverage(&terms, &format!("{} {}", document.name, chunk.text));
        if !keyword_is_relevant(&terms, coverage) {
            continue;
        }
        keyword_snippets
            .entry(chunk.document_id.clone())
            .or_insert_with(|| chunk.text.clone());
        let normalized = (hit.score / max_score).clamp(0.0, 1.0);
        let candidate = candidates.entry(chunk.document_id).or_default();
        if normalized > candidate.keyword {
            candidate.keyword = normalized;
            candidate.snippet = Some(chunk.text);
        }
    }

    if request.mode != SearchMode::Keyword {
        let query_embedding = embedder.embed_query(&request.query);
        let semantic_chunks =
            storage.semantic_chunk_search(&query_embedding, extension, candidate_limit);
        if semantic_chunks.is_empty() {
            for (document_id, score) in
                storage.semantic_search(&query_embedding, extension, candidate_limit)
            {
                if score >= RAW_SEMANTIC_MIN_SCORE {
                    candidates.entry(document_id).or_default().semantic = score;
                }
            }
        } else {
            for (document_id, chunk_id, score) in semantic_chunks {
                if score < RAW_SEMANTIC_MIN_SCORE {
                    continue;
                }
                let Some(chunk) = storage.chunk(chunk_id)? else {
                    continue;
                };
                if !by_id.contains_key(chunk.document_id.as_str()) {
                    continue;
                }
                let candidate = candidates.entry(document_id).or_default();
                if score > candidate.semantic {
                    candidate.semantic = score;
                    candidate.semantic_snippet = Some(chunk.text);
                }
            }
        }
    }

    let query_lower = request.query.to_lowercase();
    if request.mode != SearchMode::Keyword {
        for document in &documents {
            if document.name.to_lowercase().contains(&query_lower) {
                candidates.entry(document.id.clone()).or_default();
            }
        }
    }
    let mut ranked = candidates
        .into_iter()
        .filter_map(|(document_id, candidate)| {
            let document = by_id.get(document_id.as_str())?;
            let filename_boost = if document.name.to_lowercase().contains(&query_lower) {
                0.20
            } else {
                0.0
            };
            let semantic_text = candidate
                .semantic_snippet
                .as_deref()
                .unwrap_or(document.name.as_str());
            let lexical_relevance =
                term_coverage(&terms, &format!("{} {semantic_text}", document.name));
            let calibrated_semantic = candidate.semantic * 0.72 + lexical_relevance * 0.28;
            if candidate.keyword == 0.0
                && filename_boost == 0.0
                && candidate.semantic < SEMANTIC_ONLY_MIN_SCORE
            {
                return None;
            }
            let score = match request.mode {
                SearchMode::Keyword => candidate.keyword,
                SearchMode::Semantic => calibrated_semantic + candidate.keyword * 0.18,
                SearchMode::Hybrid => candidate.keyword * 0.62 + calibrated_semantic * 0.38,
            } + filename_boost;
            let threshold = match request.mode {
                SearchMode::Keyword => 0.01,
                SearchMode::Semantic => SEMANTIC_RESULT_MIN_SCORE,
                SearchMode::Hybrid => 0.06,
            };
            let matched_text = match request.mode {
                SearchMode::Keyword => candidate.snippet,
                SearchMode::Semantic => candidate.semantic_snippet.or(candidate.snippet),
                SearchMode::Hybrid if candidate.semantic > candidate.keyword => {
                    candidate.semantic_snippet.or(candidate.snippet)
                }
                SearchMode::Hybrid => candidate.snippet.or(candidate.semantic_snippet),
            };
            (score >= threshold).then_some((document_id, score, matched_text))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(request.limit.clamp(1, 100));

    let mut results = Vec::with_capacity(ranked.len());
    for (document_id, score, matched_text) in ranked {
        let document = by_id[document_id.as_str()];
        let text = match matched_text.or_else(|| keyword_snippets.get(&document_id).cloned()) {
            Some(text) => text,
            None => best_semantic_snippet(storage, document)?,
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

fn keyword_is_relevant(terms: &[String], coverage: f32) -> bool {
    if terms.is_empty() {
        return false;
    }
    let minimum_matches = (terms.len() as f32 * KEYWORD_MIN_COVERAGE)
        .ceil()
        .min(4.0)
        .max(terms.len().min(2) as f32);
    (coverage * terms.len() as f32).round() >= minimum_matches
}

fn term_coverage(query_terms: &[String], text: &str) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let matched = query_terms
        .iter()
        .filter(|term| {
            if term.is_ascii() {
                lower
                    .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .any(|word| word == term.as_str())
            } else {
                lower.contains(term.as_str())
            }
        })
        .count();
    matched as f32 / query_terms.len() as f32
}

fn best_semantic_snippet(storage: &Storage, document: &StoredDocument) -> Result<String> {
    Ok(storage
        .chunks_for_document_limited(&document.id, SEMANTIC_SNIPPET_SCAN_LIMIT)?
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

    #[test]
    fn lexical_relevance_rewards_matching_semantic_snippets() {
        let terms = query_terms("软件需求规格说明书");
        let related = term_coverage(&terms, "项目软件需求规格说明书模板");
        let unrelated = term_coverage(&terms, "新能源功率预测模型");
        assert!(related > 0.9);
        assert_eq!(unrelated, 0.0);
    }

    #[test]
    fn weak_keywords_do_not_become_relevant_by_normalization() {
        let terms = query_terms("深海潜水装备保养步骤");
        let coverage = term_coverage(&terms, "软件安装步骤");
        assert!(!keyword_is_relevant(&terms, coverage));
        assert!(!keyword_is_relevant(&[], 0.0));
        let terms = query_terms("时间");
        assert!(keyword_is_relevant(
            &terms,
            term_coverage(&terms, "完成时间")
        ));
        let terms = query_terms("电");
        assert!(keyword_is_relevant(
            &terms,
            term_coverage(&terms, "电力系统")
        ));
        let terms = query_terms("数据备份 恢复");
        assert!(keyword_is_relevant(
            &terms,
            term_coverage(&terms, "数据库备份和恢复")
        ));
    }

    #[test]
    fn keyword_search_filters_noise_without_losing_short_queries() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let document = crate::model::PreparedDocument {
            id: "backup".to_owned(),
            root: "/test".to_owned(),
            path: "/test/备份.txt".to_owned(),
            name: "数据库备份.txt".to_owned(),
            extension: "txt".to_owned(),
            modified_ms: 1,
            size: 100,
            chunks: vec!["数据库备份操作步骤与恢复时间，每日定时执行。".to_owned()],
            embedding: crate::embedding::fallback_embed("数据库备份"),
        };
        let chunks = storage.upsert_document(&document).unwrap();
        index
            .replace_document(&document.id, &document.name, &chunks)
            .unwrap();
        index.commit().unwrap();
        let engine = EmbeddingEngine::new(directory.path().join("unused-model"));
        for (query, expected) in [
            ("深海潜水装备保养步骤", 0),
            ("", 0),
            ("时间", 1),
            ("备", 1),
            ("数据库备份", 1),
        ] {
            let request = SearchRequest {
                query: query.to_owned(),
                mode: SearchMode::Keyword,
                extension: None,
                limit: 50,
            };
            assert_eq!(
                search(&storage, &index, &engine, &request).unwrap().len(),
                expected,
                "{query}"
            );
        }
    }

    #[test]
    fn coverage_preserves_token_boundaries_and_long_question_recall() {
        assert_eq!(term_coverage(&query_terms("cat"), "concatenate"), 0.0);
        assert_eq!(term_coverage(&query_terms("cat"), "CAT.txt"), 1.0);
        assert_eq!(term_coverage(&query_terms("备份"), "备 份"), 0.0);
        let terms = (0..30)
            .map(|index| format!("term{index}"))
            .collect::<Vec<_>>();
        assert!(keyword_is_relevant(&terms, 4.0 / 30.0));
        assert!(!keyword_is_relevant(&terms, 1.0 / 30.0));
    }
}
