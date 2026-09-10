use anyhow::{anyhow, Context, Result};
use encoding_rs::{GB18030, UTF_16BE, UTF_16LE};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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
            decode_text_bytes(&bytes)
        }
        "pdf" => {
            if pdf_has_encrypt_marker(path)? {
                return Err(anyhow!("加密 PDF 不支持，请解密后重新索引"));
            }
            pdf_extract::extract_text(path)
                .with_context(|| format!("PDF 解析失败: {}", path.display()))?
        }
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

fn decode_text_bytes(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        return String::from_utf8_lossy(&bytes[3..]).into_owned();
    }
    if bytes.starts_with(&[0xff, 0xfe]) {
        return UTF_16LE.decode(&bytes[2..]).0.into_owned();
    }
    if bytes.starts_with(&[0xfe, 0xff]) {
        return UTF_16BE.decode(&bytes[2..]).0.into_owned();
    }
    if let Ok(value) = std::str::from_utf8(bytes) {
        return value.to_owned();
    }
    GB18030.decode(bytes).0.into_owned()
}

fn pdf_has_encrypt_marker(path: &Path) -> Result<bool> {
    const WINDOW: u64 = 128 * 1024;
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    let start = length.saturating_sub(WINDOW);
    file.seek(SeekFrom::Start(start))?;
    let mut tail = Vec::with_capacity((length - start) as usize);
    file.read_to_end(&mut tail)?;
    Ok(tail
        .windows(b"/Encrypt".len())
        .any(|window| window == b"/Encrypt"))
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
        let extracted = match kind {
            OoxmlKind::Word => word_xml_text(&xml)?,
            _ => xml_text(&xml)?,
        };
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

fn word_xml_text(xml: &str) -> Result<String> {
    let mut reader = quick_xml::NsReader::from_str(xml);
    let mut output = String::new();
    let mut in_text = false;
    loop {
        let (namespace, event) = reader.read_resolved_event()?;
        let is_word = matches!(namespace, quick_xml::name::ResolveResult::Bound(namespace)
            if matches!(namespace.as_ref(),
                b"http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                | b"http://purl.oclc.org/ooxml/wordprocessingml/main"));
        match event {
            Event::Start(start) if is_word && start.local_name().as_ref() == b"t" => {
                in_text = true;
            }
            Event::Text(text) if in_text => output.push_str(&text.unescape()?),
            Event::End(end) if is_word => match end.local_name().as_ref() {
                b"t" => in_text = false,
                b"p" | b"tr" => output.push('\n'),
                b"tc" => output.push(' '),
                _ => {}
            },
            Event::Empty(start) if is_word => match start.local_name().as_ref() {
                b"tab" => output.push(' '),
                b"br" | b"cr" => output.push('\n'),
                _ => {}
            },
            Event::Eof => break,
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
    use std::io::Write;

    #[test]
    fn extracts_xml_text_without_tags() {
        let value = xml_text("<w:p><w:r><w:t>本地搜索</w:t></w:r><w:r><w:t> MVP</w:t></w:r></w:p>")
            .unwrap();
        assert!(value.contains("本地搜索"));
        assert!(value.contains("MVP"));
    }

    #[test]
    fn word_fields_preserve_display_text_and_run_boundaries() {
        let xml = r#"<x:document xmlns:x="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <x:p><x:r><x:instrText>TOC HYPERLINK PAGEREF _Toc123</x:instrText></x:r>
          <x:hyperlink><x:r><x:t>目录标题</x:t></x:r></x:hyperlink>
          <x:fldSimple x:instr="PAGE"><x:r><x:t>1</x:t></x:r></x:fldSimple></x:p>
          <x:p><x:r><x:t>数据</x:t></x:r><x:r><x:t>备份</x:t></x:r>
          <x:r><x:tab/><x:t xml:space="preserve"> TOC 是正文术语 &amp; 示例</x:t><x:br/><x:t>下一行</x:t></x:r></x:p>
        </x:document>"#;
        let text = word_xml_text(xml).unwrap();
        assert!(text.contains("目录标题1\n"));
        assert!(text.contains("数据备份"));
        assert!(text.contains("TOC 是正文术语 & 示例\n下一行"));
        assert!(!text.contains("PAGEREF"));
        assert!(!text.contains("_Toc123"));
    }

    #[test]
    fn docx_extraction_discards_instructions_but_keeps_field_results() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fields.docx");
        let mut archive = zip::ZipWriter::new(File::create(&path).unwrap());
        archive
            .start_file(
                "word/document.xml",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
        archive
            .write_all(
                br#"<document xmlns="http://purl.oclc.org/ooxml/wordprocessingml/main">
          <p><r><instrText>PAGEREF _Toc123</instrText></r><r><t>7</t></r></p>
          <p><r><t>Data</t></r><r><t xml:space="preserve"> backup</t></r></p>
        </document>"#,
            )
            .unwrap();
        archive.finish().unwrap();
        assert_eq!(extract_text(&path).unwrap(), "7\nData backup");
    }

    #[test]
    fn decodes_utf8_and_legacy_chinese_text() {
        let value = "本地文档搜索，支持中文";
        assert_eq!(decode_text_bytes(value.as_bytes()), value);
        let (encoded, _, had_errors) = GB18030.encode(value);
        assert!(!had_errors);
        assert_eq!(decode_text_bytes(&encoded), value);
    }

    #[test]
    fn decodes_utf16_text_with_bom() {
        let value = "本地文档搜索";
        let mut bytes = vec![0xff, 0xfe];
        bytes.extend(value.encode_utf16().flat_map(u16::to_le_bytes));
        assert_eq!(decode_text_bytes(&bytes), value);
    }

    #[test]
    fn rejects_corrupt_office_archive() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("broken.docx");
        std::fs::write(&path, b"not a zip archive").unwrap();
        let error = extract_text(&path).unwrap_err().to_string();
        assert!(error.contains("无效") || error.to_lowercase().contains("invalid"));
    }

    #[test]
    fn identifies_encrypted_pdf_before_parsing() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("encrypted.pdf");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"%PDF-1.7\n1 0 obj << /Encrypt 2 0 R >>\n%%EOF")
            .unwrap();
        let error = extract_text(&path).unwrap_err().to_string();
        assert!(error.contains("加密 PDF"));
    }

    #[cfg(unix)]
    #[test]
    fn reports_permission_denied_files() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("private.txt");
        std::fs::write(&path, "secret").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = extract_text(&path);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        if let Err(error) = result {
            assert_eq!(
                crate::indexer::classify_failure(&format!("{error:#}")),
                "permission"
            );
        }
    }
}
