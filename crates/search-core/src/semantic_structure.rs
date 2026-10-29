use crate::model::PreparedDocument;
use std::collections::HashSet;

const MAX_ANCHORS: usize = 32;
const MAX_ANCHOR_CHARS: usize = 80;

pub fn is_descriptive_anchor(text: &str) -> bool {
    text.split('、').any(|field| {
        !matches!(
            field.trim().to_lowercase().as_str(),
            "" | "页码"
                | "行号"
                | "序号"
                | "编号"
                | "内容"
                | "id"
                | "page"
                | "row"
                | "index"
                | "content"
        )
    })
}

pub fn document_anchors(document: &PreparedDocument) -> Vec<(usize, String)> {
    let mut anchors = Vec::new();
    let mut seen = HashSet::new();
    for (position, chunk) in document.chunks.iter().enumerate() {
        let lines = chunk.lines().collect::<Vec<_>>();
        for (line_index, line) in lines.iter().enumerate() {
            if (position > 0 && line_index == 0)
                || (position + 1 < document.chunks.len() && line_index + 1 == lines.len())
            {
                continue;
            }
            let anchor = table_schema(line).or_else(|| section_title(line));
            if let Some(anchor) = anchor {
                if anchor
                    .chars()
                    .filter(|character| character.is_alphanumeric())
                    .count()
                    < 4
                {
                    continue;
                }
                let key = anchor.split_whitespace().collect::<String>().to_lowercase();
                if !key.is_empty() && seen.insert(key) {
                    anchors.push((position, anchor));
                    if anchors.len() == MAX_ANCHORS {
                        return anchors;
                    }
                }
            }
        }
    }
    anchors
}

fn table_schema(line: &str) -> Option<String> {
    let columns = line
        .split('|')
        .skip(1)
        .filter_map(|column| column.trim().split_once(':'))
        .collect::<Vec<_>>();
    if columns.len() < 2 {
        return None;
    }
    let header_row = columns.iter().all(|(key, _)| {
        !key.is_empty() && key.chars().all(|character| character.is_ascii_uppercase())
    });
    let mut seen = HashSet::new();
    let fields = columns
        .iter()
        .map(|(key, value)| if header_row { *value } else { *key }.trim())
        .filter(|field| !field.is_empty() && field.chars().count() <= 32)
        .filter(|field| seen.insert(field.to_lowercase()))
        .collect::<Vec<_>>();
    let schema = fields.join("、");
    (fields.len() >= 2 && schema.chars().count() <= MAX_ANCHOR_CHARS).then_some(schema)
}

fn section_title(line: &str) -> Option<String> {
    let line = line.trim();
    let title = if line.starts_with('#') {
        line.trim_start_matches('#').trim()
    } else if let Some(rest) = line.strip_prefix('第') {
        let number_end = rest.find(|character: char| {
            !matches!(
                character,
                '一' | '二'
                    | '三'
                    | '四'
                    | '五'
                    | '六'
                    | '七'
                    | '八'
                    | '九'
                    | '十'
                    | '百'
                    | '千'
                    | '零'
                    | '〇'
                    | '两'
            ) && !character.is_ascii_digit()
        })?;
        if number_end == 0 {
            return None;
        }
        let rest = &rest[number_end..];
        let unit = rest.chars().next()?;
        if !matches!(unit, '章' | '节' | '条' | '篇') {
            return None;
        }
        rest[unit.len_utf8()..].trim_start_matches([' ', '：', ':', '、', '.', '．'])
    } else {
        return None;
    };
    let length = title.chars().count();
    (length >= 2
        && length <= MAX_ANCHOR_CHARS
        && !title.contains(['。', '；', ';', '，', ',', '！', '？']))
    .then(|| title.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigation_only_schemas_are_not_topic_evidence() {
        assert!(!is_descriptive_anchor("页码、行号、内容"));
        assert!(!is_descriptive_anchor("ID、Page、Row、Content"));
        assert!(is_descriptive_anchor(
            "排名、Institution Name、院校名称、地区"
        ));
        assert!(is_descriptive_anchor("编号、类型、评分"));
        assert!(is_descriptive_anchor("货款结算"));
    }

    #[test]
    fn headings_remove_structural_numbers_not_content() {
        assert_eq!(section_title("第九条 ：货款结算"), Some("货款结算".into()));
        assert_eq!(section_title("第十二章 系统恢复"), Some("系统恢复".into()));
        assert_eq!(
            section_title("## GPU 运行环境"),
            Some("GPU 运行环境".into())
        );
        assert_eq!(
            section_title("第一条：目的需方委托供方，双方签订协议。"),
            None
        );
        assert_eq!(section_title("第一个例子"), None);
        assert_eq!(section_title("普通正文"), None);
    }

    #[test]
    fn table_headers_and_rows_have_identical_schema_without_cell_values() {
        let expected = Some("排名、Institution Name、院校名称、地区".to_owned());
        assert_eq!(
            table_schema(
                "工作表 1 第 2 行 | A: 排名 | B: Institution Name | C: 院校名称 | D: 地区"
            ),
            expected
        );
        assert_eq!(table_schema("工作表 1 第 3 行 | 排名: 1 | Institution Name: MIT | 院校名称: 麻省理工学院 | 地区: 美国"), expected);
        assert_eq!(
            table_schema("工作表 1 第 1 行 | 排名: QS 2026世界大学排名"),
            None
        );
        assert_eq!(table_schema("没有表格"), None);
    }

    #[test]
    fn anchors_are_bounded_and_keep_their_source_chunk() {
        let document = PreparedDocument {
            id: "test".into(),
            root: "/test".into(),
            path: "/test/test.txt".into(),
            name: "test.txt".into(),
            extension: "txt".into(),
            modified_ms: 1,
            size: 1,
            embedding: Vec::new(),
            chunks: vec![
                "正文".into(),
                (0..100)
                    .map(|number| format!("# 标题{number}\n# 标题{number}"))
                    .collect(),
            ],
        };
        let anchors = document_anchors(&document);
        assert_eq!(anchors.len(), MAX_ANCHORS);
        assert!(anchors.iter().all(|(position, _)| *position == 1));
    }

    #[test]
    fn incomplete_chunk_edges_and_ambiguous_short_headings_are_not_anchors() {
        let document = PreparedDocument {
            id: "test".into(),
            root: "/test".into(),
            path: "/test/test.pdf".into(),
            name: "test.pdf".into(),
            extension: "pdf".into(),
            modified_ms: 1,
            size: 1,
            embedding: Vec::new(),
            chunks: vec![
                "第六条：附件\n第七条：品质保证及不良担".into(),
                "及不良担保责任\n第九条：货款结算\n正文".into(),
            ],
        };
        assert_eq!(document_anchors(&document), vec![(1, "货款结算".into())]);
    }
}
