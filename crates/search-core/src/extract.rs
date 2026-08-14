use anyhow::{anyhow, Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "pdf", "docx", "xlsx", "pptx", "txt", "md", "csv", "json", "log", "rst",
];

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| SUPPORTED_EXTENSIONS.contains(&value.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn extract_text(path: &Path) -> Result<String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let text = match extension.as_str() {
        "txt" | "md" | "csv" | "json" | "log" | "rst" => {
            let bytes =
                std::fs::read(path).with_context(|| format!("无法读取 {}", path.display()))?;
            String::from_utf8_lossy(&bytes).into_owned()
        }
        "pdf" => pdf_extract::extract_text(path)
            .with_context(|| format!("PDF 解析失败: {}", path.display()))?,
        "docx" => extract_ooxml(path, OoxmlKind::Word)?,
        "pptx" => extract_ooxml(path, OoxmlKind::PowerPoint)?,
        "xlsx" => extract_ooxml(path, OoxmlKind::Excel)?,
        _ => return Err(anyhow!("暂不支持 .{extension} 格式")),
    };

    let normalized = normalize_text(&text);
    if normalized.is_empty() {
        return Err(anyhow!("未提取到可索引文本"));
    }
    Ok(normalized)
}

enum OoxmlKind {
    Word,
    PowerPoint,
    Excel,
}

fn extract_ooxml(path: &Path, kind: OoxmlKind) -> Result<String> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file).context("无效的 Office Open XML 文件")?;
    let mut names = (0..archive.len())
        .filter_map(|index| {
            archive
                .by_index(index)
                .ok()
                .map(|entry| entry.name().to_owned())
        })
        .filter(|name| match kind {
            OoxmlKind::Word => name == "word/document.xml",
            OoxmlKind::PowerPoint => name.starts_with("ppt/slides/slide") && name.ends_with(".xml"),
            OoxmlKind::Excel => {
                name == "xl/sharedStrings.xml"
                    || (name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml"))
            }
        })
        .collect::<Vec<_>>();
    names.sort_by_key(|name| natural_number(name));

    let mut output = String::new();
    for name in names {
        let mut entry = archive.by_name(&name)?;
        let mut xml = String::new();
        entry.read_to_string(&mut xml)?;
        let extracted = xml_text(&xml)?;
        if !extracted.is_empty() {
            output.push_str(&extracted);
            output.push('\n');
        }
    }
    Ok(output)
}

fn natural_number(value: &str) -> u64 {
    value
        .chars()
        .filter(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or(0)
}

fn xml_text(xml: &str) -> Result<String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut output = String::new();
    loop {
        match reader.read_event() {
            Ok(Event::Text(text)) => {
                let value = text.unescape()?;
                if !value.trim().is_empty() {
                    if !output.is_empty() {
                        output.push(' ');
                    }
                    output.push_str(value.trim());
                }
            }
            Ok(Event::End(end)) if matches!(end.name().as_ref(), b"p" | b"tr" | b"row") => {
                output.push('\n');
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(error.into()),
            _ => {}
        }
    }
    Ok(output)
}

fn normalize_text(value: &str) -> String {
    value
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_xml_text_without_tags() {
        let value = xml_text("<w:p><w:r><w:t>本地搜索</w:t></w:r><w:r><w:t> MVP</w:t></w:r></w:p>")
            .unwrap();
        assert!(value.contains("本地搜索"));
        assert!(value.contains("MVP"));
    }
}
