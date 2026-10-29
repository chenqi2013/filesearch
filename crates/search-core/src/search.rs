use crate::embedding::{query_terms, EmbeddingEngine, ASCII_PREFIX_MIN_LENGTH};
use crate::model::{IndexedDocument, SearchMode, SearchRequest, SearchResult, StoredDocument};
use crate::storage::{SemanticChunkMatch, Storage};
use crate::text_index::TextIndex;
use anyhow::Result;
use std::collections::HashMap;

#[derive(Default)]
struct Candidate {
    keyword: f32,
    short_cjk_coverage: f32,
    short_concept_confirmed: bool,
    semantic: f32,
    semantic_relevance: f32,
    semantic_confirmation: f32,
    semantic_coverage: f32,
    semantic_precision: bool,
    semantic_minimum: Option<f32>,
    cross_query_support: bool,
    keyword_fallback: f32,
    fallback_snippet: Option<String>,
    topic: f32,
    structure_relevance: f32,
    structure_snippet: Option<String>,
    snippet: Option<String>,
    semantic_snippet: Option<String>,
    local_snippet: Option<String>,
    local_snippet_score: f32,
    local_snippet_joint: f32,
    local_snippet_coverage: f32,
    alternative_snippet: Option<String>,
    alternative_snippet_score: f32,
    alternative_snippet_joint: f32,
    alternative_snippet_coverage: f32,
}

const SEMANTIC_SNIPPET_SCAN_LIMIT: usize = 1;
const RAW_SEMANTIC_MIN_SCORE: f32 = 0.24;
const SEMANTIC_RESULT_MIN_SCORE: f32 = 0.30;
const KEYWORD_MIN_COVERAGE: f32 = 0.25;
const SEMANTIC_ONLY_ABSOLUTE_MIN_SCORE: f32 = 0.40;
const SEMANTIC_AGREEMENT_MIN_SCORE: f32 = 0.40;
const SEMANTIC_ONLY_RELATIVE_RATIO: f32 = 0.84;
const SEMANTIC_KEYWORD_MIN_COVERAGE: f32 = 0.30;
const SHORT_CJK_SEMANTIC_MIN_SCORE: f32 = 0.30;
const SHORT_CJK_MIN_COVERAGE: f32 = 1.0;
const SHORT_CJK_RELEVANCE_BOOST: f32 = 0.10;
const UNANCHORED_SEMANTIC_MIN_SCORE: f32 = 0.43;
const LOCAL_PASSAGE_MIN_SCORE: f32 = 0.20;
const SNIPPET_SEMANTIC_IMPROVEMENT: f32 = 0.10;
const SNIPPET_LEXICAL_RETENTION: f32 = 0.75;

impl Candidate {
    fn consider_snippet(&mut self, text: &str, semantic: f32, confirmation: f32, lexical: f32) {
        let joint = (semantic * confirmation).sqrt();
        let lexical_score = joint * 0.50 + lexical * 0.50;
        if lexical_score > self.local_snippet_score {
            self.local_snippet_score = lexical_score;
            self.local_snippet_joint = joint;
            self.local_snippet_coverage = lexical;
            self.local_snippet = Some(text.to_owned());
        }
        let semantic_score = joint * 0.80 + lexical * 0.20;
        if semantic_score > self.alternative_snippet_score {
            self.alternative_snippet_score = semantic_score;
            self.alternative_snippet_joint = joint;
            self.alternative_snippet_coverage = lexical;
            self.alternative_snippet = Some(text.to_owned());
        }
    }

    fn select_snippet(&mut self) {
        // Keep lexical localization for close alternatives. A different passage must
        // have both absolute support and a clear same-passage semantic improvement.
        // Retain most of the query's lexical evidence so a broad semantic topic
        // cannot replace a more specific clause. Document admission is unchanged.
        if self.alternative_snippet_joint >= SEMANTIC_AGREEMENT_MIN_SCORE
            && self.alternative_snippet_joint
                >= self.local_snippet_joint + SNIPPET_SEMANTIC_IMPROVEMENT
            && self.alternative_snippet_coverage
                >= self.local_snippet_coverage * SNIPPET_LEXICAL_RETENTION
        {
            self.local_snippet = self.alternative_snippet.take();
        }
    }
}

pub fn search(
    storage: &Storage,
    text_index: &TextIndex,
    embedder: &EmbeddingEngine,
    request: &SearchRequest,
) -> Result<Vec<SearchResult>> {
    if request.query.trim().is_empty() {
        return Ok(Vec::new());
    }
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
    let weights = if request.mode == SearchMode::Keyword {
        vec![1.0; terms.len()]
    } else {
        text_index.query_term_weights(&terms)?
    };
    let candidate_limit = (request.limit.clamp(1, 100) * 20).clamp(200, 1_000);

    let hits = text_index.search(&request.query, candidate_limit)?;
    let keyword_chunks = hits.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>();
    let mut keyword_scores = HashMap::new();
    let mut semantic_only_min_score = SEMANTIC_ONLY_ABSOLUTE_MIN_SCORE;
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
        let lexical_text = format!("{} {}", document.name, chunk.text);
        let normalized = (hit.score / max_score).clamp(0.0, 1.0);
        keyword_scores.insert(hit.chunk_id, normalized);
        let coverage = term_coverage(&terms, &lexical_text);
        if !keyword_is_relevant(&terms, coverage) {
            continue;
        }
        keyword_snippets
            .entry(chunk.document_id.clone())
            .or_insert_with(|| chunk.text.clone());
        let fallback = if request.mode == SearchMode::Keyword {
            0.0
        } else {
            precise_keyword_relevance(
                normalized,
                coverage,
                precise_lexical_match(&request.query, &terms, &lexical_text),
            )
        };
        let candidate = candidates.entry(chunk.document_id).or_default();
        if fallback > candidate.keyword_fallback {
            candidate.keyword_fallback = fallback;
            candidate.fallback_snippet = Some(chunk.text.clone());
        }
        if normalized > candidate.keyword {
            candidate.keyword = normalized;
            candidate.snippet = Some(chunk.text);
        }
    }

    if request.mode != SearchMode::Keyword {
        let query_embedding = embedder.embed_query(&request.query);
        let confirmation = embedder.embed_query(&format!("关于{}的资料", request.query.trim()));
        let mut semantic_chunks = storage.semantic_passage_matches(
            &query_embedding,
            &confirmation,
            extension,
            candidate_limit,
            8,
            &keyword_chunks,
        );
        let topics = storage
            .semantic_search(&query_embedding, extension, documents.len())
            .into_iter()
            .collect::<HashMap<_, _>>();
        if semantic_chunks.is_empty() {
            let document_matches =
                storage.semantic_search(&query_embedding, extension, candidate_limit);
            let best_score = document_matches
                .iter()
                .map(|(_, score)| *score)
                .fold(0.0f32, f32::max);
            semantic_only_min_score =
                (best_score * SEMANTIC_ONLY_RELATIVE_RATIO).max(SEMANTIC_ONLY_ABSOLUTE_MIN_SCORE);
            let confirmations = storage
                .semantic_search(&confirmation, extension, documents.len())
                .into_iter()
                .collect::<HashMap<_, _>>();
            for (document_id, score) in document_matches {
                if score >= RAW_SEMANTIC_MIN_SCORE {
                    let candidate = candidates.entry(document_id.clone()).or_default();
                    candidate.semantic = score;
                    candidate.semantic_confirmation =
                        confirmations.get(&document_id).copied().unwrap_or_default();
                }
            }
        } else {
            let best_score = semantic_chunks
                .iter()
                .map(|hit| hit.score)
                .fold(0.0f32, f32::max);
            semantic_only_min_score =
                (best_score * SEMANTIC_ONLY_RELATIVE_RATIO).max(SEMANTIC_ONLY_ABSOLUTE_MIN_SCORE);
            let instructed_query = embedder.embed_query(&format!(
                "Instruct: Given a query, retrieve documents that answer the query\nQuery: {}",
                request.query.trim()
            ));
            let instructed_chunks = storage.semantic_passage_matches(
                &instructed_query,
                &confirmation,
                extension,
                candidate_limit,
                8,
                &keyword_chunks,
            );
            let instructed_best = instructed_chunks
                .iter()
                .map(|hit| hit.score)
                .fold(0.0f32, f32::max);
            let instructed_minimum = (instructed_best * SEMANTIC_ONLY_RELATIVE_RATIO)
                .max(SEMANTIC_ONLY_ABSOLUTE_MIN_SCORE);
            let corroborated = cross_query_winner(&semantic_chunks, &instructed_chunks);
            let original_len = semantic_chunks.len();
            semantic_chunks.extend(instructed_chunks);
            for (position, hit) in semantic_chunks.into_iter().enumerate() {
                let original_view = position < original_len;
                if hit.score < LOCAL_PASSAGE_MIN_SCORE {
                    continue;
                }
                let Some(document) = by_id.get(hit.document_id.as_str()) else {
                    continue;
                };
                let local_text = snippet_with_weights(&hit.text, &request.query, &terms, &weights);
                let lexical_text = format!("{} {}", document.name, hit.text);
                let coverage = term_coverage(&terms, &lexical_text);
                let short_coverage = if original_view {
                    short_cjk_coverage(&request.query, &lexical_text)
                } else {
                    0.0
                };
                let confirmed = original_view
                    && short_concept_confirmed(
                        &request.query,
                        &hit.text,
                        hit.score,
                        hit.confirmation,
                    );
                let mut window = Candidate {
                    cross_query_support: corroborated.as_deref() == Some(hit.document_id.as_str())
                        && !original_view,
                    semantic_minimum: Some(if original_view {
                        semantic_only_min_score
                    } else {
                        instructed_minimum
                    }),
                    semantic: hit.score,
                    semantic_confirmation: hit.confirmation,
                    semantic_coverage: coverage,
                    semantic_precision: precise_lexical_match(
                        &request.query,
                        &terms,
                        &lexical_text,
                    ),
                    short_cjk_coverage: short_coverage,
                    short_concept_confirmed: confirmed,
                    semantic_relevance: hit.score * 0.72
                        + coverage * 0.28
                        + if short_coverage >= SHORT_CJK_MIN_COVERAGE || confirmed {
                            SHORT_CJK_RELEVANCE_BOOST
                        } else {
                            0.0
                        },
                    ..Candidate::default()
                };
                if !body_is_supported(&window, semantic_only_min_score) {
                    window.semantic_relevance = 0.0;
                }
                let candidate = candidates.entry(hit.document_id.clone()).or_default();
                if window.semantic_relevance > candidate.semantic_relevance {
                    candidate.semantic = window.semantic;
                    candidate.semantic_minimum = window.semantic_minimum;
                    candidate.cross_query_support = window.cross_query_support;
                    candidate.semantic_confirmation = hit.confirmation;
                    candidate.semantic_precision = window.semantic_precision;
                    candidate.semantic_coverage = coverage;
                    candidate.short_cjk_coverage = short_coverage;
                    candidate.short_concept_confirmed = confirmed;
                    candidate.semantic_relevance = window.semantic_relevance;
                    candidate.semantic_snippet = Some(hit.text.clone());
                }
                candidate.consider_snippet(
                    &hit.text,
                    hit.score,
                    hit.confirmation,
                    weighted_coverage(&terms, &weights, &local_text),
                );
                if let Some(keyword) = keyword_scores.get(&hit.chunk_id) {
                    let fallback = local_keyword_relevance(
                        *keyword,
                        coverage,
                        hit.score,
                        hit.confirmation,
                        weighted_coverage(&terms, &weights, &local_text),
                        &terms,
                        &local_text,
                    );
                    if fallback > candidate.keyword_fallback {
                        candidate.keyword_fallback = fallback;
                        candidate.fallback_snippet = Some(local_text);
                    }
                }
            }
        }
        for (document_id, text, anchor_score, body_score) in
            storage.semantic_anchor_search(&query_embedding, extension)
        {
            let score = structure_relevance(&request.query, anchor_score, body_score);
            if score <= 0.0 || !by_id.contains_key(document_id.as_str()) {
                continue;
            }
            let candidate = candidates.entry(document_id).or_default();
            if score > candidate.structure_relevance {
                candidate.structure_relevance = score;
                candidate.structure_snippet = Some(text);
            }
        }
        for (id, candidate) in &mut candidates {
            candidate.topic = topics.get(id).copied().unwrap_or_default();
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
    let fingerprints = storage.content_fingerprints();
    for (document_id, candidate) in &mut candidates {
        if let Some(document) = by_id
            .get(document_id.as_str())
            .filter(|_| candidate.semantic_relevance == 0.0 && candidate.semantic > 0.0)
        {
            let semantic_text = candidate
                .semantic_snippet
                .as_deref()
                .or(candidate.snippet.as_deref())
                .unwrap_or(document.name.as_str());
            let coverage = term_coverage(&terms, &format!("{} {semantic_text}", document.name));
            let short_cjk_coverage = short_cjk_coverage(
                &request.query,
                &format!("{} {semantic_text}", document.name),
            );
            candidate.short_cjk_coverage = candidate.short_cjk_coverage.max(short_cjk_coverage);
            candidate.semantic_precision = precise_lexical_match(
                &request.query,
                &terms,
                &format!("{} {semantic_text}", document.name),
            );
            candidate.semantic_coverage = coverage;
            candidate.short_concept_confirmed = short_concept_confirmed(
                &request.query,
                semantic_text,
                candidate.semantic,
                candidate.semantic_confirmation,
            );
            candidate.semantic_relevance = candidate.semantic * 0.72
                + coverage * 0.28
                + if short_cjk_coverage >= SHORT_CJK_MIN_COVERAGE
                    || candidate.short_concept_confirmed
                {
                    SHORT_CJK_RELEVANCE_BOOST
                } else {
                    0.0
                };
        }
    }
    for candidate in candidates.values_mut() {
        finalize_semantic_candidate(candidate, semantic_only_min_score);
        candidate.select_snippet();
    }
    let keyword_ranks = channel_ranks(
        candidates
            .iter()
            .filter(|(_, candidate)| candidate.keyword_fallback > 0.0)
            .map(|(id, candidate)| (id.clone(), candidate.keyword_fallback))
            .collect(),
        &fingerprints,
    );
    let semantic_ranks = channel_ranks(
        candidates
            .iter()
            .filter(|(_, candidate)| {
                (candidate.semantic >= RAW_SEMANTIC_MIN_SCORE
                    || candidate.structure_relevance > 0.0)
                    && candidate.semantic_relevance >= SEMANTIC_RESULT_MIN_SCORE
            })
            .map(|(id, candidate)| (id.clone(), candidate.semantic_relevance))
            .collect(),
        &fingerprints,
    );
    let mut ranked = candidates
        .into_iter()
        .filter_map(|(document_id, candidate)| {
            let document = by_id.get(document_id.as_str())?;
            let filename_boost = if document.name.to_lowercase().contains(&query_lower) {
                0.20
            } else {
                0.0
            };
            let semantic_evidence = candidate.semantic_relevance >= SEMANTIC_RESULT_MIN_SCORE;
            let keyword_fallback = semantic_keyword_fallback(&candidate);
            let supported = match request.mode {
                SearchMode::Keyword => candidate.keyword > 0.0,
                SearchMode::Semantic => {
                    semantic_evidence || keyword_fallback > 0.0 || filename_boost > 0.0
                }
                SearchMode::Hybrid => {
                    candidate.keyword_fallback > 0.0 || semantic_evidence || filename_boost > 0.0
                }
            };
            if !supported {
                return None;
            }
            let score = match request.mode {
                SearchMode::Keyword => candidate.keyword,
                SearchMode::Semantic => candidate.semantic_relevance.max(keyword_fallback),
                SearchMode::Hybrid => reciprocal_rank_score(
                    keyword_ranks.get(&document_id).copied(),
                    semantic_ranks.get(&document_id).copied(),
                ),
            } + filename_boost;
            let threshold = match request.mode {
                SearchMode::Keyword => 0.01,
                SearchMode::Semantic => SEMANTIC_RESULT_MIN_SCORE,
                SearchMode::Hybrid => 0.06,
            };
            let matched_text = match request.mode {
                SearchMode::Keyword => candidate.snippet,
                _ if candidate.local_snippet.is_some() => candidate.local_snippet,
                SearchMode::Semantic if candidate.semantic_relevance >= keyword_fallback => {
                    candidate.semantic_snippet.or(candidate.snippet)
                }
                SearchMode::Semantic => candidate.fallback_snippet.or(candidate.semantic_snippet),
                SearchMode::Hybrid
                    if semantic_ranks
                        .get(&document_id)
                        .copied()
                        .unwrap_or(usize::MAX)
                        < keyword_ranks
                            .get(&document_id)
                            .copied()
                            .unwrap_or(usize::MAX) =>
                {
                    candidate.semantic_snippet.or(candidate.snippet)
                }
                SearchMode::Hybrid => candidate.fallback_snippet.or(candidate.semantic_snippet),
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
    let mut seen = std::collections::HashSet::new();
    ranked.retain(|(id, _, _)| seen.insert(fingerprints.get(id).unwrap_or(id).clone()));
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
            snippet: if request.mode == SearchMode::Keyword {
                snippet(&text, &request.query)
            } else {
                snippet_with_weights(&text, &request.query, &terms, &weights)
            },
            duplicates: documents
                .iter()
                .filter(|other| {
                    other.id != document.id
                        && fingerprints.get(&document.id).is_some_and(|fingerprint| {
                            fingerprints.get(&other.id) == Some(fingerprint)
                        })
                })
                .map(|other| IndexedDocument {
                    id: other.id.clone(),
                    path: other.path.clone(),
                    name: other.name.clone(),
                    extension: other.extension.clone(),
                    modified_ms: other.modified_ms,
                    size: other.size,
                })
                .collect(),
            score: ((score.clamp(0.0, 1.0) * 100.0).round()) / 100.0,
        });
    }
    Ok(results)
}

fn cross_query_winner(
    original: &[SemanticChunkMatch],
    instructed: &[SemanticChunkMatch],
) -> Option<String> {
    let winner = |passages: &[SemanticChunkMatch], minimum: f32| {
        let mut scores: HashMap<&str, f32> = HashMap::new();
        for passage in passages {
            let value = scores.entry(&passage.document_id).or_default();
            *value = value.max(passage.score);
        }
        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0))
        });
        let (identity, score) = *ranked.first()?;
        let (_, runner_up) = *ranked.get(1)?;
        (score >= minimum && runner_up < score * SEMANTIC_ONLY_RELATIVE_RATIO)
            .then(|| identity.to_owned())
    };
    let original_winner = winner(original, RAW_SEMANTIC_MIN_SCORE)?;
    (winner(instructed, SEMANTIC_ONLY_ABSOLUTE_MIN_SCORE)? == original_winner)
        .then_some(original_winner)
}

fn channel_ranks(
    mut scores: Vec<(String, f32)>,
    fingerprints: &HashMap<String, String>,
) -> HashMap<String, usize> {
    scores.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut groups = HashMap::new();
    let mut ranks = HashMap::new();
    for (id, _) in scores {
        let next_rank = groups.len() + 1;
        let key = fingerprints.get(&id).unwrap_or(&id).clone();
        let rank = *groups.entry(key).or_insert(next_rank);
        ranks.insert(id, rank);
    }
    ranks
}

fn reciprocal_rank_score(keyword: Option<usize>, semantic: Option<usize>) -> f32 {
    const RRF_OFFSET: f32 = 60.0;
    const KEYWORD_WEIGHT: f32 = 0.25;
    const SEMANTIC_WEIGHT: f32 = 0.75;
    let contribution = |rank: Option<usize>| {
        rank.map_or(0.0, |rank| (RRF_OFFSET + 1.0) / (RRF_OFFSET + rank as f32))
    };
    contribution(keyword) * KEYWORD_WEIGHT + contribution(semantic) * SEMANTIC_WEIGHT
}

fn semantic_keyword_fallback(candidate: &Candidate) -> f32 {
    candidate.keyword_fallback
        + if candidate.keyword_fallback > 0.0 {
            candidate.topic * 0.20
        } else {
            0.0
        }
}

fn precise_keyword_relevance(keyword: f32, coverage: f32, precise: bool) -> f32 {
    if keyword <= 0.0 || !precise || coverage < SEMANTIC_KEYWORD_MIN_COVERAGE {
        return 0.0;
    }
    keyword * 0.20 + coverage * 0.20 + 0.05
}

fn local_keyword_relevance(
    keyword: f32,
    coverage: f32,
    semantic: f32,
    confirmation: f32,
    weighted: f32,
    terms: &[String],
    text: &str,
) -> f32 {
    if !keyword_is_relevant(terms, term_coverage(terms, text))
        || coverage < KEYWORD_MIN_COVERAGE
        || semantic < RAW_SEMANTIC_MIN_SCORE
        || confirmation < RAW_SEMANTIC_MIN_SCORE
        || weighted < 0.60
    {
        return 0.0;
    }
    keyword * 0.20 + coverage * 0.20 + 0.05
}

fn weighted_coverage(terms: &[String], weights: &[f32], text: &str) -> f32 {
    let total = weights.iter().sum::<f32>();
    if total <= 0.0 {
        return term_coverage(terms, text);
    }
    let lower = text.to_lowercase();
    terms
        .iter()
        .zip(weights)
        .filter_map(|(term, weight)| {
            (term_matches(term, &lower)
                || (term.is_ascii()
                    && term.len() > 4
                    && term.strip_suffix('s').is_some_and(|singular| {
                        lower
                            .split(|character: char| {
                                !character.is_ascii_alphanumeric() && character != '_'
                            })
                            .any(|word| word == singular)
                    })))
            .then_some(weight)
        })
        .sum::<f32>()
        / total
}

fn body_is_supported(candidate: &Candidate, semantic_only_min_score: f32) -> bool {
    let semantic_only_min_score = candidate
        .semantic_minimum
        .unwrap_or(semantic_only_min_score);
    let joint = (candidate.semantic * candidate.semantic_confirmation).sqrt();
    let agreement = candidate.semantic >= semantic_only_min_score
        || (candidate.semantic >= SEMANTIC_RESULT_MIN_SCORE && joint >= semantic_only_min_score);
    let grounded = candidate.semantic_coverage > 0.0 || joint >= UNANCHORED_SEMANTIC_MIN_SCORE;
    (agreement
        && (candidate.semantic_precision || (joint >= SEMANTIC_AGREEMENT_MIN_SCORE && grounded)))
        || (candidate.cross_query_support
            && candidate.semantic >= SEMANTIC_ONLY_ABSOLUTE_MIN_SCORE
            && candidate.semantic_confirmation >= RAW_SEMANTIC_MIN_SCORE)
        || (candidate.semantic >= SHORT_CJK_SEMANTIC_MIN_SCORE
            && candidate.short_cjk_coverage >= SHORT_CJK_MIN_COVERAGE)
        || candidate.short_concept_confirmed
}

fn short_concept_confirmed(query: &str, text: &str, score: f32, confirmation: f32) -> bool {
    let characters = query.trim().chars().collect::<Vec<_>>();
    characters.len() == 2
        && characters.iter().all(|character| {
            matches!(*character as u32, 0x3400..=0x9fff | 0xf900..=0xfaff)
                && text.contains(*character)
        })
        && score >= SHORT_CJK_SEMANTIC_MIN_SCORE
        && confirmation >= (score * 0.90).min(0.35)
}

fn finalize_semantic_candidate(candidate: &mut Candidate, semantic_only_min_score: f32) {
    if !body_is_supported(candidate, semantic_only_min_score) {
        candidate.semantic_relevance = 0.0;
    }
    let structure_score = if candidate.structure_relevance >= SEMANTIC_RESULT_MIN_SCORE {
        candidate.structure_relevance + 0.10
    } else {
        0.0
    };
    if structure_score > candidate.semantic_relevance {
        candidate.semantic_relevance = structure_score;
        candidate.semantic_snippet = candidate.structure_snippet.clone();
    }
    if candidate.semantic_relevance > 0.0 {
        candidate.semantic_relevance += candidate.topic.max(0.0) * 0.20;
    }
}

fn precise_lexical_match(query: &str, terms: &[String], text: &str) -> bool {
    let query = query.trim().to_lowercase();
    let text = text.to_lowercase();
    if query.is_empty() {
        return false;
    }
    if query.is_ascii() {
        return term_coverage(terms, &text) >= 0.75;
    }
    if text.contains(&query) {
        return true;
    }
    let characters = query.chars().collect::<Vec<_>>();
    if characters.len() > 4 && term_coverage(terms, &text) < 0.50 {
        return false;
    }
    let minimum = if characters.len() <= 4 {
        characters.len()
    } else {
        3
    };
    characters.windows(minimum).any(|window| {
        window
            .iter()
            .all(|character| matches!(*character as u32, 0x3400..=0x9fff | 0xf900..=0xfaff))
            && text.contains(&window.iter().collect::<String>())
    })
}

fn structure_relevance(query: &str, anchor_score: f32, body_score: f32) -> f32 {
    let short_concept = query
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count()
        <= 3;
    let body_minimum = if short_concept { 0.24 } else { 0.30 };
    if anchor_score < 0.40 || body_score < body_minimum {
        return 0.0;
    }
    let joint = anchor_score * 0.65 + body_score * 0.35;
    if joint < 0.36 {
        return 0.0;
    }
    joint * 0.85
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
        .filter(|term| term_matches(term, &lower))
        .count();
    matched as f32 / query_terms.len() as f32
}

fn term_matches(term: &str, lower: &str) -> bool {
    if term.is_ascii() {
        lower
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .any(|word| {
                word == term || (term.len() >= ASCII_PREFIX_MIN_LENGTH && word.starts_with(term))
            })
    } else {
        lower.contains(term)
    }
}

fn short_cjk_coverage(query: &str, text: &str) -> f32 {
    let characters = query
        .trim()
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<std::collections::HashSet<_>>();
    if !(2..=4).contains(&characters.len())
        || characters
            .iter()
            .any(|character| !matches!(*character as u32, 0x3400..=0x9fff | 0xf900..=0xfaff))
    {
        return 0.0;
    }
    if text.to_lowercase().contains(&query.trim().to_lowercase()) {
        1.0
    } else {
        0.0
    }
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
    let terms = query_terms(query);
    snippet_with_weights(text, query, &terms, &vec![1.0; terms.len()])
}

fn snippet_with_weights(text: &str, query: &str, terms: &[String], weights: &[f32]) -> String {
    const WINDOW: usize = 260;
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= WINDOW + 6 {
        return text.to_owned();
    }
    let lower = text.to_lowercase();
    let query_lower = query.to_lowercase();
    let start = if let Some(position) = lower.find(&query_lower) {
        lowercase_char_position(text, position).saturating_sub(70)
    } else {
        terms
            .iter()
            .flat_map(|term| {
                let value = if term.is_ascii() && term.len() > 4 {
                    term.strip_suffix('s').unwrap_or(term)
                } else {
                    term
                };
                lower.match_indices(value).map(|(position, _)| position)
            })
            .map(|position| lowercase_char_position(text, position).saturating_sub(70))
            .fold((0, -1.0f32), |best, start| {
                let end = (start + WINDOW).min(chars.len());
                let value = chars[start..end].iter().collect::<String>();
                let score = weighted_coverage(terms, weights, &value);
                if score > best.1 {
                    (start, score)
                } else {
                    best
                }
            })
            .0
    };
    let start = start.min(chars.len());
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

fn lowercase_char_position(text: &str, byte_position: usize) -> usize {
    let mut offset = 0;
    for (position, character) in text.chars().enumerate() {
        if offset >= byte_position {
            return position;
        }
        offset += character.to_lowercase().map(char::len_utf8).sum::<usize>();
    }
    text.chars().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_can_escape_lexical_overlap_with_clear_local_semantic_support() {
        let mut candidate = Candidate::default();
        candidate.consider_snippet("unrelated repeated query terms", 0.30, 0.30, 0.80);
        candidate.consider_snippet("answer phrased differently", 0.41, 0.41, 0.65);
        assert_eq!(
            candidate.local_snippet.as_deref(),
            Some("unrelated repeated query terms")
        );
        candidate.semantic_relevance = 0.53;
        candidate.select_snippet();
        assert_eq!(
            candidate.local_snippet.as_deref(),
            Some("answer phrased differently")
        );
        assert_eq!(candidate.semantic_relevance, 0.53);
    }

    #[test]
    fn snippet_keeps_lexical_evidence_for_close_or_weak_alternatives() {
        for (original, alternative) in [(0.32, 0.40), (0.20, 0.35)] {
            let mut candidate = Candidate::default();
            candidate.consider_snippet("specific evidence", original, original, 1.0);
            candidate.consider_snippet("broad topic", alternative, alternative, 0.50);
            candidate.select_snippet();
            assert_eq!(
                candidate.local_snippet.as_deref(),
                Some("specific evidence")
            );
        }
    }

    #[test]
    fn snippet_support_cannot_be_borrowed_from_a_different_passage() {
        let mut candidate = Candidate::default();
        candidate.consider_snippet("lexical evidence", 0.30, 0.30, 1.0);
        candidate.consider_snippet("raw query only", 0.90, 0.10, 0.40);
        candidate.consider_snippet("confirmation only", 0.10, 0.90, 0.40);
        candidate.select_snippet();
        assert_eq!(candidate.local_snippet.as_deref(), Some("lexical evidence"));
    }

    #[test]
    fn semantic_topic_cannot_replace_a_clause_with_much_more_specific_evidence() {
        let mut candidate = Candidate::default();
        candidate.consider_snippet("specific clause", 0.30, 0.30, 0.90);
        candidate.consider_snippet(
            "semantically similar but different clause",
            0.50,
            0.50,
            0.50,
        );
        candidate.select_snippet();
        assert_eq!(candidate.local_snippet.as_deref(), Some("specific clause"));
    }

    #[test]
    fn cross_query_support_requires_distinct_document_agreement_and_a_strong_instruction() {
        let passage = |document: &str, score| SemanticChunkMatch {
            chunk_id: 1,
            document_id: document.into(),
            text: String::new(),
            score,
            confirmation: score,
        };
        let original = [
            passage("first", 0.31),
            passage("first", 0.30),
            passage("other", 0.18),
        ];
        assert_eq!(
            cross_query_winner(&original, &[passage("first", 0.43), passage("other", 0.24)]),
            Some("first".into())
        );
        for instructed in [
            vec![passage("other", 0.43), passage("first", 0.24)],
            vec![passage("first", 0.43), passage("other", 0.42)],
            vec![passage("first", 0.35), passage("other", 0.20)],
            vec![passage("first", 0.43)],
            Vec::new(),
        ] {
            assert!(cross_query_winner(&original, &instructed).is_none());
        }
        assert!(cross_query_winner(
            &[passage("first", 0.31), passage("other", 0.30)],
            &[passage("first", 0.43), passage("other", 0.24)],
        )
        .is_none());
        assert!(cross_query_winner(
            &[passage("first", 0.20), passage("other", 0.10)],
            &[passage("first", 0.43), passage("other", 0.24)],
        )
        .is_none());
    }

    #[test]
    fn corroborated_evidence_still_requires_a_supported_local_window() {
        let mut candidate = Candidate {
            cross_query_support: true,
            semantic: 0.43,
            semantic_confirmation: 0.25,
            ..Candidate::default()
        };
        assert!(body_is_supported(&candidate, 0.40));
        candidate.semantic_confirmation = 0.10;
        assert!(!body_is_supported(&candidate, 0.40));
        candidate.semantic_confirmation = 0.25;
        candidate.semantic = 0.30;
        assert!(!body_is_supported(&candidate, 0.40));
        candidate.cross_query_support = false;
        candidate.semantic = 0.43;
        assert!(!body_is_supported(&candidate, 0.40));
    }

    #[test]
    fn structure_recall_requires_both_anchor_and_local_body_evidence() {
        assert_eq!(structure_relevance("概念", 0.9, 0.2), 0.0);
        assert_eq!(structure_relevance("概念", 0.39, 0.7), 0.0);
        assert_eq!(structure_relevance("概念", 0.4, 0.24), 0.0);
        assert_eq!(structure_relevance("更具体的问题", 0.8, 0.28), 0.0);
        let score = structure_relevance("概念", 0.45, 0.32);
        assert!(score >= SEMANTIC_RESULT_MIN_SCORE);
        let mut candidate = Candidate {
            structure_relevance: score,
            ..Candidate::default()
        };
        finalize_semantic_candidate(&mut candidate, 0.8);
        assert!(candidate.semantic_relevance >= SEMANTIC_RESULT_MIN_SCORE);
    }

    #[test]
    fn structure_recall_returns_source_text_without_changing_keyword_mode() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let engine = EmbeddingEngine::new(directory.path().join("missing-model"));
        let query = "概念";
        let vector = engine.embed_query(query);
        let mut orthogonal = vec![0.0; vector.len()];
        let basis = vector.iter().position(|value| value.abs() < 0.001).unwrap();
        orthogonal[basis] = 1.0;
        let body_vector = vector
            .iter()
            .zip(orthogonal)
            .map(|(value, basis)| value * 0.28 + basis * (1.0f32 - 0.28f32.powi(2)).sqrt())
            .collect::<Vec<_>>();
        let document = crate::model::PreparedDocument {
            id: "structured".into(),
            root: "/test".into(),
            path: "/test/intro.txt".into(),
            name: "intro.txt".into(),
            extension: "txt".into(),
            modified_ms: 1,
            size: 100,
            chunks: vec!["这是章节对应的原始正文，用于返回可核对的片段。".into()],
            embedding: Vec::new(),
        };
        let chunks = storage.upsert_document(&document).unwrap();
        index
            .replace_document(&document.id, &document.name, &chunks)
            .unwrap();
        index.commit().unwrap();
        storage
            .replace_chunk_embeddings(&[(chunks[0].id, document.chunks[0].clone(), body_vector)])
            .unwrap();
        let request = |mode| SearchRequest {
            query: query.into(),
            mode,
            extension: None,
            limit: 10,
        };
        assert!(
            search(&storage, &index, &engine, &request(SearchMode::Semantic))
                .unwrap()
                .is_empty()
        );
        storage
            .replace_anchor_embeddings(&[(chunks[0].id, "章节标题".into(), vector)])
            .unwrap();
        for mode in [SearchMode::Semantic, SearchMode::Hybrid] {
            let results = search(&storage, &index, &engine, &request(mode)).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "structured");
            assert!(results[0].snippet.contains("原始正文"));
        }
        assert!(
            search(&storage, &index, &engine, &request(SearchMode::Keyword))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn snippet_keeps_result_bounded() {
        let text = format!("{}目标词{}", "前".repeat(400), "后".repeat(400));
        let value = snippet(&text, "目标词");
        assert!(value.chars().count() <= 266);
        assert!(value.contains("目标词"));
    }

    #[test]
    fn snippet_prefers_a_local_cluster_over_the_first_generic_match() {
        let text = format!(
            "文件{}月结时使用入库单和对账单，原始凭据需要保留。{}",
            "说明".repeat(200),
            "后文".repeat(200)
        );
        let terms = vec!["文件".into(), "月结".into(), "对账".into(), "凭据".into()];
        let value = snippet_with_weights(&text, "文件月结对账凭据", &terms, &[1.0, 5.0, 5.0, 5.0]);
        assert!(value.contains("月结"));
        assert!(value.contains("对账单"));
        assert!(value.contains("凭据"));
        assert!(value.chars().count() <= 266);
    }

    #[test]
    fn local_keyword_support_requires_same_window_semantics_and_specific_terms() {
        let terms = query_terms("月结对账需要哪些凭据");
        let text = "实行月结，提交对账单";
        let coverage = term_coverage(&terms, text);
        let relevance = |semantic, confirmation, weighted| {
            local_keyword_relevance(
                1.0,
                coverage,
                semantic,
                confirmation,
                weighted,
                &terms,
                text,
            )
        };
        assert!(relevance(0.27, 0.28, 0.8) >= SEMANTIC_RESULT_MIN_SCORE);
        assert_eq!(relevance(0.20, 0.28, 0.8), 0.0);
        assert_eq!(relevance(0.27, 0.20, 0.8), 0.0);
        assert_eq!(relevance(0.27, 0.28, 0.3), 0.0);
        assert_eq!(
            local_keyword_relevance(1.0, 1.0, 0.9, 0.9, 1.0, &terms, "凭据"),
            0.0
        );
    }

    #[test]
    fn paired_query_agreement_can_support_a_paraphrase_without_lowering_the_floor() {
        let candidate = |semantic, semantic_confirmation| Candidate {
            semantic,
            semantic_confirmation,
            semantic_coverage: 0.1,
            ..Candidate::default()
        };
        assert!(body_is_supported(&candidate(0.35, 0.60), 0.40));
        assert!(body_is_supported(&candidate(0.48, 0.36), 0.40));
        assert!(!body_is_supported(&candidate(0.80, 0.10), 0.40));
        assert!(!body_is_supported(&candidate(0.20, 0.90), 0.40));
        assert!(!body_is_supported(&candidate(0.35, 0.35), 0.40));
        let mut ungrounded = Candidate {
            semantic: 0.42,
            semantic_confirmation: 0.42,
            ..Candidate::default()
        };
        assert!(!body_is_supported(&ungrounded, 0.40));
        ungrounded.semantic = 0.50;
        ungrounded.semantic_confirmation = 0.50;
        assert!(body_is_supported(&ungrounded, 0.40));
    }

    #[test]
    fn semantic_evidence_beyond_the_display_crop_is_not_discarded() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let engine = EmbeddingEngine::new(directory.path().join("missing-model"));
        let query = "alpha beta gamma delta epsilon";
        let query_vector = engine.embed_query(query);
        let confirmation = engine.embed_query(&format!("关于{query}的资料"));
        let overlap = crate::embedding::cosine(&query_vector, &confirmation);
        let raw = 0.42;
        let confirmed = 0.25;
        let raw_weight = (raw - confirmed * overlap) / (1.0 - overlap * overlap);
        let confirmed_weight = (confirmed - raw * overlap) / (1.0 - overlap * overlap);
        let mut vector = query_vector
            .iter()
            .zip(&confirmation)
            .map(|(left, right)| left * raw_weight + right * confirmed_weight)
            .collect::<Vec<_>>();
        let norm = crate::embedding::cosine(&vector, &vector);
        assert!(norm < 1.0);
        let unused_dimension = query_vector
            .iter()
            .zip(&confirmation)
            .position(|(left, right)| *left == 0.0 && *right == 0.0)
            .unwrap();
        vector[unused_dimension] = (1.0 - norm).sqrt();
        assert!((crate::embedding::cosine(&query_vector, &vector) - raw).abs() < 0.001);
        assert!((crate::embedding::cosine(&confirmation, &vector) - confirmed).abs() < 0.001);

        let text = format!("alpha beta {} gamma delta epsilon", "背景".repeat(300));
        let terms = query_terms(query);
        assert!(precise_lexical_match(query, &terms, &text));
        assert!(!precise_lexical_match(
            query,
            &terms,
            &snippet(&text, query)
        ));
        for (id, body) in [("evidence", text), ("unrelated", "无关正文".into())] {
            let document = crate::model::PreparedDocument {
                id: id.into(),
                root: "/test".into(),
                path: format!("/test/{id}.txt"),
                name: format!("{id}.txt"),
                extension: "txt".into(),
                modified_ms: 1,
                size: body.len() as u64,
                chunks: vec![body.clone()],
                embedding: Vec::new(),
            };
            let chunks = storage.upsert_document(&document).unwrap();
            storage
                .replace_chunk_embeddings(&[(chunks[0].id, body, vector.clone())])
                .unwrap();
        }
        assert!(index.search(query, 10).unwrap().is_empty());
        for mode in [SearchMode::Semantic, SearchMode::Hybrid] {
            let results = search(
                &storage,
                &index,
                &engine,
                &SearchRequest {
                    query: query.into(),
                    mode,
                    extension: None,
                    limit: 10,
                },
            )
            .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "evidence");
            assert!(results[0].snippet.contains("gamma delta epsilon"));
        }
    }

    #[test]
    fn plural_snippet_terms_keep_word_boundaries() {
        let terms = vec!["annotations".into(), "phone".into()];
        assert_eq!(
            weighted_coverage(&terms, &[8.0, 1.0], "Annotation tools on phones"),
            1.0
        );
        assert_eq!(
            weighted_coverage(&terms, &[8.0, 1.0], "annotationless telephones"),
            0.0
        );
    }

    #[test]
    fn snippet_handles_lowercase_expansion_without_losing_the_match() {
        let text = format!("{} target {}", "İ".repeat(300), "İ".repeat(300));
        for query in ["target", "target additional"] {
            let value = snippet(&text, query);
            assert!(value.contains("target"));
            assert!(value.chars().count() <= 266);
        }
    }

    #[test]
    fn structured_recall_displays_query_specific_content_instead_of_the_header() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let engine = EmbeddingEngine::new(directory.path().join("missing-model"));
        let query = "新加坡的大学能排到多少名";
        let vector = engine.embed_query(query);
        let document = crate::model::PreparedDocument {
            id: "ranking".into(),
            root: "/test".into(),
            path: "/test/ranking.xlsx".into(),
            name: "ranking.xlsx".into(),
            extension: "xlsx".into(),
            modified_ms: 1,
            size: 100,
            chunks: vec![
                "排名、院校名称、地区：麻省理工学院，美国".into(),
                "排名: 8 | 院校名称: 新加坡国立大学 | 地区: 新加坡".into(),
            ],
            embedding: Vec::new(),
        };
        let chunks = storage.upsert_document(&document).unwrap();
        index
            .replace_document(&document.id, &document.name, &chunks)
            .unwrap();
        index.commit().unwrap();
        storage
            .replace_chunk_embeddings(
                &chunks
                    .iter()
                    .map(|chunk| (chunk.id, chunk.text.clone(), vector.clone()))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        storage
            .replace_anchor_embeddings(&[(chunks[0].id, "排名、院校名称、地区".into(), vector)])
            .unwrap();
        for mode in [SearchMode::Semantic, SearchMode::Hybrid] {
            let results = search(
                &storage,
                &index,
                &engine,
                &SearchRequest {
                    query: query.into(),
                    mode,
                    extension: None,
                    limit: 10,
                },
            )
            .unwrap();
            assert_eq!(results[0].id, "ranking");
            assert!(results[0].snippet.contains("新加坡国立大学"));
            assert!(!results[0].snippet.contains("麻省理工学院"));
        }
    }

    #[test]
    fn reciprocal_ranks_reward_agreement_without_duplicate_crowding() {
        let fingerprints = HashMap::from([
            ("first".to_owned(), "same".to_owned()),
            ("copy".to_owned(), "same".to_owned()),
        ]);
        let ranks = channel_ranks(
            vec![
                ("first".to_owned(), 0.9),
                ("copy".to_owned(), 0.8),
                ("other".to_owned(), 0.7),
            ],
            &fingerprints,
        );
        assert_eq!(ranks["first"], 1);
        assert_eq!(ranks["copy"], 1);
        assert_eq!(ranks["other"], 2);
        assert!(reciprocal_rank_score(Some(2), Some(2)) > reciprocal_rank_score(Some(1), None));
        assert!(reciprocal_rank_score(None, Some(1)) > reciprocal_rank_score(Some(1), None));
        assert_eq!(reciprocal_rank_score(None, None), 0.0);
        assert_eq!(reciprocal_rank_score(Some(1), Some(1)), 1.0);
    }

    #[test]
    fn semantic_mode_keeps_strong_keyword_evidence_without_broad_fallbacks() {
        assert!(precise_keyword_relevance(0.9, 0.5, true) >= SEMANTIC_RESULT_MIN_SCORE);
        assert_eq!(precise_keyword_relevance(1.0, 0.2, true), 0.0);
        assert_eq!(precise_keyword_relevance(1.0, 1.0, false), 0.0);
    }

    #[test]
    fn lexical_precision_rejects_scattered_generic_terms_but_keeps_prefixes() {
        for (query, text, expected) in [
            ("天气预报", "天气寒冷，需要提前报告", false),
            ("显卡驱动安装", "神秘力量驱动着物体，随后安装了程序", false),
            ("世界一流大学", "精神世界一片宁静", false),
            ("坐办公室的岗位", "办公室四级主任科员及以下", false),
            ("坐办公室的岗位", "岗位名称：办公室四级主任科员及以下", true),
            ("Praesent pu", "Praesent pulvinar", true),
            ("cat", "concatenate", false),
            ("", "anything", false),
        ] {
            assert_eq!(
                precise_lexical_match(query, &query_terms(query), text),
                expected,
                "{query}"
            );
        }
    }

    #[test]
    fn weak_keyword_or_structure_cannot_borrow_unsupported_body_score() {
        let mut candidate = Candidate {
            keyword: 1.0,
            semantic: 0.41,
            semantic_relevance: 0.8,
            semantic_confirmation: 0.20,
            ..Candidate::default()
        };
        finalize_semantic_candidate(&mut candidate, 0.40);
        assert_eq!(candidate.semantic_relevance, 0.0);
        assert_eq!(semantic_keyword_fallback(&candidate), 0.0);
        candidate.semantic_relevance = 0.8;
        candidate.structure_relevance = 0.31;
        candidate.structure_snippet = Some("locally supported body".into());
        candidate.semantic_snippet = Some("unsupported body".into());
        finalize_semantic_candidate(&mut candidate, 0.40);
        assert!((candidate.semantic_relevance - 0.41).abs() < 0.001);
        assert_eq!(
            candidate.semantic_snippet.as_deref(),
            Some("locally supported body")
        );
    }

    #[test]
    fn two_character_concepts_require_local_characters_and_semantic_confirmation() {
        let text = "逾期付款需要赔偿损失";
        assert!(short_concept_confirmed("赔款", text, 0.32, 0.30));
        assert!(!short_concept_confirmed("赔款", text, 0.32, 0.20));
        assert!(!short_concept_confirmed("赔款", "采购款", 0.32, 0.40));
        assert!(!short_concept_confirmed("赔款", text, 0.25, 0.40));
        assert!(!short_concept_confirmed(
            "天气预报",
            "天气寒冷，需要提前报告",
            0.32,
            0.40
        ));
    }

    #[test]
    fn short_cjk_coverage_requires_contiguous_query_for_full_support() {
        assert_eq!(
            short_cjk_coverage("赔款", "逾期付款需要支付违约金并承担赔偿责任"),
            0.0
        );
        assert_eq!(short_cjk_coverage("赔款", "采购款应当按时支付"), 0.0);
        assert_eq!(short_cjk_coverage("赔款", "支付赔款"), 1.0);
        assert_eq!(
            short_cjk_coverage("适合全家一起看的轻松片子", "全家观看"),
            0.0
        );
        assert!(body_is_supported(
            &Candidate {
                semantic: 0.31,
                short_cjk_coverage: 1.0,
                ..Candidate::default()
            },
            0.40
        ));
        assert!(!body_is_supported(
            &Candidate {
                semantic: 0.29,
                short_cjk_coverage: 1.0,
                ..Candidate::default()
            },
            0.40
        ));
        assert!(!body_is_supported(
            &Candidate {
                semantic: 0.31,
                short_cjk_coverage: 0.5,
                ..Candidate::default()
            },
            0.40
        ));
    }

    #[test]
    fn semantic_search_keeps_matching_terms_from_unsampled_content() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let document = crate::model::PreparedDocument {
            id: "jobs".to_owned(),
            root: "/test".to_owned(),
            path: "/test/重庆考公职位表.xlsx".to_owned(),
            name: "重庆考公职位表.xlsx".to_owned(),
            extension: "xlsx".to_owned(),
            modified_ms: 1,
            size: 100,
            chunks: vec!["岗位名称: 办公室四级主任科员及以下 | 招录人数: 2".to_owned()],
            embedding: Vec::new(),
        };
        let chunks = storage.upsert_document(&document).unwrap();
        index
            .replace_document(&document.id, &document.name, &chunks)
            .unwrap();
        index.commit().unwrap();
        let engine = EmbeddingEngine::new(directory.path().join("missing-model"));
        let results = search(
            &storage,
            &index,
            &engine,
            &SearchRequest {
                query: "坐办公室的岗位".to_owned(),
                mode: SearchMode::Semantic,
                extension: None,
                limit: 10,
            },
        )
        .unwrap();
        assert_eq!(
            results.first().map(|result| result.id.as_str()),
            Some("jobs")
        );
    }

    #[test]
    #[ignore = "loads the bundled RWKV model"]
    fn customer_semantic_cases_recall_expected_documents() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let engine = EmbeddingEngine::new(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/models/embedding-rwkv-tiny"),
        );
        let cases = [
            (
                "contract",
                "2024-03供应商合作协议.pdf",
                "供应商合作协议：采购方逾期付款时，应按未付金额支付违约金并承担赔偿责任。",
                "赔款",
            ),
            (
                "jobs",
                "重庆考公职位表.xlsx",
                "工作表 招录职位 | 岗位名称: 办公室四级主任科员及以下 | 招录人数: 2",
                "坐办公室的岗位",
            ),
            (
                "movies",
                "经典电影数据.xlsx",
                "电影名: 欢乐家庭 | 类型: 喜剧 动画 | 适合人群: 全家观看 | 氛围: 轻松",
                "适合全家一起看的轻松片子",
            ),
            (
                "songs",
                "KTV热歌.txt",
                "歌单分类: 经典怀旧 | 内容: 往年流行金曲和经典老歌",
                "以前很火的老歌",
            ),
        ];
        for (id, name, text, _) in &cases {
            let extension = name.rsplit_once('.').unwrap().1;
            let vector = engine.embed_passages(&[text.to_string()]).remove(0);
            let document = crate::model::PreparedDocument {
                id: (*id).to_owned(),
                root: "/test".to_owned(),
                path: format!("/test/{name}"),
                name: (*name).to_owned(),
                extension: extension.to_owned(),
                modified_ms: 1,
                size: text.len() as u64,
                chunks: vec![(*text).to_owned()],
                embedding: vector.clone(),
            };
            let chunks = storage.upsert_document(&document).unwrap();
            index
                .replace_document(&document.id, &document.name, &chunks)
                .unwrap();
            storage
                .replace_chunk_embeddings(&[(chunks[0].id, (*text).to_owned(), vector)])
                .unwrap();
            storage.update_document_embeddings(&[document]).unwrap();
        }
        index.commit().unwrap();
        for (expected_id, _, _, query) in cases {
            let results = search(
                &storage,
                &index,
                &engine,
                &SearchRequest {
                    query: query.to_owned(),
                    mode: SearchMode::Semantic,
                    extension: None,
                    limit: 10,
                },
            )
            .unwrap();
            assert_eq!(
                results.first().map(|result| result.id.as_str()),
                Some(expected_id),
                "query={query:?}, results={:?}",
                results
                    .iter()
                    .map(|result| (&result.id, result.score))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    #[ignore = "loads the bundled RWKV model"]
    fn pdf_contract_semantic_recall_uses_production_passage_shape() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        let engine = EmbeddingEngine::new(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/models/embedding-rwkv-tiny"),
        );
        let name = "2024-03供应商合作协议.pdf";
        let text = "2024年3月供应商合作协议\n甲方应在收到合格发票后十五个工作日内支付采购款。\n甲方逾期付款的，每逾期一日，应按未付金额的千分之一支付违约金。\n因一方违约造成对方损失的，违约方应承担相应赔偿责任。\n双方应优先协商解决履约争议。";
        let passage = format!("{name}\n{text}");
        let vector = engine.embed_passages(&[passage]).remove(0);
        let document = crate::model::PreparedDocument {
            id: "contract".to_owned(),
            root: "/test".to_owned(),
            path: format!("/test/{name}"),
            name: name.to_owned(),
            extension: "pdf".to_owned(),
            modified_ms: 1,
            size: text.len() as u64,
            chunks: vec![text.to_owned()],
            embedding: vector.clone(),
        };
        let chunks = storage.upsert_document(&document).unwrap();
        index
            .replace_document(&document.id, &document.name, &chunks)
            .unwrap();
        storage
            .replace_chunk_embeddings(&[(chunks[0].id, text.to_owned(), vector)])
            .unwrap();
        storage.update_document_embeddings(&[document]).unwrap();
        index.commit().unwrap();
        let results = search(
            &storage,
            &index,
            &engine,
            &SearchRequest {
                query: "赔款".to_owned(),
                mode: SearchMode::Semantic,
                extension: None,
                limit: 10,
            },
        )
        .unwrap();
        assert_eq!(
            results.first().map(|result| result.id.as_str()),
            Some("contract"),
            "raw={:?}",
            storage
                .semantic_chunk_matches(
                    &engine.embed_query("赔款"),
                    &engine.embed_query("关于赔款的资料"),
                    None,
                    10
                )
                .iter()
                .map(|hit| (hit.score, hit.confirmation))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn duplicate_folding_preserves_paths_limit_and_extension_filter() {
        let directory = tempfile::tempdir().unwrap();
        let storage = Storage::open(&directory.path().join("search.db")).unwrap();
        let index = TextIndex::open(&directory.path().join("tantivy")).unwrap();
        for (id, extension, body) in [
            ("first", "txt", "backup restore"),
            ("copy", "txt", "backup restore"),
            ("pdf", "pdf", "backup restore"),
            ("other", "txt", "backup schedule details"),
        ] {
            let document = crate::model::PreparedDocument {
                id: id.to_owned(),
                root: "/test".to_owned(),
                path: format!("/test/{id}.{extension}"),
                name: format!("{id}.{extension}"),
                extension: extension.to_owned(),
                modified_ms: 1,
                size: 12,
                chunks: vec![body.to_owned()],
                embedding: Vec::new(),
            };
            let chunks = storage.upsert_document(&document).unwrap();
            index
                .replace_document(&document.id, &document.name, &chunks)
                .unwrap();
        }
        index.commit().unwrap();
        let engine = EmbeddingEngine::new(directory.path().join("unused-model"));
        let mut request = SearchRequest {
            query: "backup".to_owned(),
            mode: SearchMode::Keyword,
            extension: None,
            limit: 2,
        };
        let results = search(&storage, &index, &engine, &request).unwrap();
        assert_eq!(results.len(), 2);
        let grouped = results
            .iter()
            .find(|result| !result.duplicates.is_empty())
            .unwrap();
        assert_eq!(grouped.duplicates.len(), 2);
        let ids = std::iter::once(grouped.id.as_str())
            .chain(
                grouped
                    .duplicates
                    .iter()
                    .map(|document| document.id.as_str()),
            )
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            ids,
            std::collections::HashSet::from(["first", "copy", "pdf"])
        );
        request.extension = Some("txt".to_owned());
        let results = search(&storage, &index, &engine, &request).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .flat_map(|result| &result.duplicates)
            .all(|document| document.extension == "txt"));
        assert_eq!(
            results
                .iter()
                .map(|result| result.duplicates.len())
                .sum::<usize>(),
            1
        );
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
