use crate::model::{DocumentRecord, SearchMode, SearchRequest, SearchResult};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

const EMBEDDING_SIZE: usize = 256;

pub fn embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_SIZE];
    for token in tokens(text) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();
        let index = (hash as usize) % EMBEDDING_SIZE;
        let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
        vector[index] += sign;
    }
    normalize(&mut vector);
    vector
}

pub fn search(documents: &[DocumentRecord], request: &SearchRequest) -> Vec<SearchResult> {
    let query_tokens = tokens(&request.query);
    let query_embedding = embed(&request.query);
    let query_lower = request.query.to_lowercase();
    let extension_filter = request
        .extension
        .as_deref()
        .filter(|value| !value.is_empty());
    let mut results = Vec::new();

    for document in documents {
        if extension_filter.is_some_and(|extension| extension != document.extension) {
            continue;
        }
        let filename_score = if document.name.to_lowercase().contains(&query_lower) {
            0.22
        } else {
            0.0
        };
        let mut best: Option<(f32, &str)> = None;
        for chunk in &document.chunks {
            let fallback_terms;
            let chunk_terms = if chunk.terms.is_empty() {
                fallback_terms = sorted_tokens(&chunk.text);
                &fallback_terms
            } else {
                &chunk.terms
            };
            let lexical = lexical_score(&query_tokens, chunk_terms);
            let semantic = cosine(&query_embedding, &chunk.embedding).max(0.0);
            let score = match request.mode {
                SearchMode::Keyword => lexical,
                SearchMode::Semantic => semantic,
                SearchMode::Hybrid => lexical * 0.68 + semantic * 0.32,
            } + filename_score;
            let threshold = match request.mode {
                SearchMode::Keyword => 0.05,
                SearchMode::Semantic => 0.08,
                SearchMode::Hybrid => 0.10,
            };
            if score > threshold && best.is_none_or(|(current, _)| score > current) {
                best = Some((score, &chunk.text));
            }
        }
        if let Some((score, text)) = best {
            results.push(SearchResult {
                id: document.id.clone(),
                path: document.path.clone(),
                name: document.name.clone(),
                extension: document.extension.clone(),
                modified_ms: document.modified_ms,
                size: document.size,
                snippet: snippet(text, &request.query),
                score: (score.min(1.0) * 100.0).round() / 100.0,
            });
        }
    }
    results.sort_by(|left, right| right.score.total_cmp(&left.score));
    results.truncate(request.limit.clamp(1, 100));
    results
}

pub fn chunks(text: &str) -> Vec<String> {
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
        output.push(chars[start..end].iter().collect::<String>());
        if end == chars.len() {
            break;
        }
        start = end - OVERLAP;
    }
    output
}

fn lexical_score(query: &HashSet<String>, terms: &[String]) -> f32 {
    if query.is_empty() {
        return 0.0;
    }
    let matched = query
        .iter()
        .filter(|token| terms.binary_search(token).is_ok())
        .count();
    matched as f32 / query.len() as f32
}

pub fn sorted_tokens(text: &str) -> Vec<String> {
    let mut values = tokens(text).into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    values
}

fn tokens(text: &str) -> HashSet<String> {
    let lower = text.to_lowercase();
    let mut output = HashSet::new();
    let mut word = String::new();
    let mut cjk = Vec::new();
    for character in lower.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            word.push(character);
        } else {
            if word.len() > 1 {
                output.insert(std::mem::take(&mut word));
            } else {
                word.clear();
            }
            if is_cjk(character) {
                cjk.push(character);
                output.insert(character.to_string());
            } else {
                cjk.clear();
            }
            if cjk.len() >= 2 {
                output.insert(cjk[cjk.len() - 2..].iter().collect());
            }
        }
    }
    if word.len() > 1 {
        output.insert(word);
    }
    output
}

fn is_cjk(character: char) -> bool {
    matches!(character as u32, 0x3400..=0x9fff | 0xf900..=0xfaff)
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
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
    fn chinese_bigrams_rank_related_text() {
        let query = embed("本地文档搜索");
        let related = embed("Windows 本地文档智能搜索工具");
        let unrelated = embed("季度餐饮费用报表");
        assert!(cosine(&query, &related) > cosine(&query, &unrelated));
    }

    #[test]
    fn chunks_keep_overlap() {
        let input = "本".repeat(1700);
        let output = chunks(&input);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].chars().count(), 800);
    }
}
