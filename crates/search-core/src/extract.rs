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

    let mut normalized = normalize_text(&text);
    if extension == "pdf" && pdf_text_needs_ocr(&normalized) {
        match extract_pdf_ocr(path) {
            Ok(ocr_text) => {
                let ocr_text = normalize_text(&ocr_text);
                if normalized.is_empty()
                    || pdf_text_quality(&ocr_text) > pdf_text_quality(&normalized)
                {
                    normalized = ocr_text;
                }
            }
            Err(error) if normalized.is_empty() => {
                return Err(error)
                    .with_context(|| format!("PDF 没有文字层，OCR 识别失败: {}", path.display()));
            }
            Err(_) => {}
        }
    }
    if normalized.is_empty() {
        return Err(anyhow!("未提取到可索引文本"));
    }
    Ok(normalized)
}

#[cfg(windows)]
fn extract_pdf_ocr(path: &Path) -> Result<String> {
    use windows::core::HSTRING;
    use windows::Data::Pdf::{PdfDocument, PdfPageRenderOptions};
    use windows::Globalization::Language;
    use windows::Graphics::Imaging::BitmapDecoder;
    use windows::Media::Ocr::OcrEngine;
    use windows::Storage::StorageFile;
    use windows::Storage::Streams::InMemoryRandomAccessStream;

    let file_path = HSTRING::from(path.as_os_str().to_string_lossy().as_ref());
    let file = StorageFile::GetFileFromPathAsync(&file_path)?.get()?;
    let document = PdfDocument::LoadFromFileAsync(&file)?.get()?;
    let profile_engine = OcrEngine::TryCreateFromUserProfileLanguages()
        .context("Windows 未安装当前用户语言对应的 OCR 组件")?;
    let profile_language = profile_engine
        .RecognizerLanguage()?
        .LanguageTag()?
        .to_string();
    let mut engines = vec![profile_engine];
    if !profile_language.eq_ignore_ascii_case("en-US") {
        let english = Language::CreateLanguage(&HSTRING::from("en-US"))?;
        if OcrEngine::IsLanguageSupported(&english)? {
            engines.push(OcrEngine::TryCreateFromLanguage(&english)?);
        }
    }
    let max_dimension = OcrEngine::MaxImageDimension()? as f32;
    let page_count = document.PageCount()?;
    let mut output = String::new();

    for page_index in 0..page_count {
        let page = document.GetPage(page_index)?;
        let page_size = page.Size()?;
        let longest_edge = page_size.Width.max(page_size.Height).max(1.0);
        let scale = (max_dimension / longest_edge).min(2.5).max(1.0);
        let options = PdfPageRenderOptions::new()?;
        options.SetDestinationWidth((page_size.Width * scale).round() as u32)?;
        options.SetDestinationHeight((page_size.Height * scale).round() as u32)?;

        let stream = InMemoryRandomAccessStream::new()?;
        page.RenderWithOptionsToStreamAsync(&stream, &options)?
            .get()?;
        stream.Seek(0)?;
        let decoder = BitmapDecoder::CreateAsync(&stream)?.get()?;
        let bitmap = decoder.GetSoftwareBitmapAsync()?.get()?;
        let mut page_text = String::new();
        let mut best_score = i64::MIN;
        for engine in &engines {
            let result = engine.RecognizeAsync(&bitmap)?.get()?;
            let candidate = result.Text()?.to_string();
            let score = ocr_text_score(&candidate);
            if score > best_score {
                best_score = score;
                page_text = candidate;
            }
        }
        if !page_text.trim().is_empty() {
            if !output.is_empty() {
                output.push_str("\n\n");
            }
            output.push_str(page_text.trim());
        }
    }

    if output.is_empty() {
        return Err(anyhow!("Windows OCR 未识别到可索引文本"));
    }
    Ok(output)
}

#[cfg(windows)]
fn ocr_text_score(text: &str) -> i64 {
    let mut latin = 0i64;
    let mut cjk = 0i64;
    let mut other_letters = 0i64;
    let mut digits = 0i64;
    let mut symbols = 0i64;

    for character in text.chars() {
        if character.is_ascii_alphabetic() {
            latin += 1;
        } else if matches!(character, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}') {
            cjk += 1;
        } else if character.is_alphabetic() {
            other_letters += 1;
        } else if character.is_ascii_digit() {
            digits += 1;
        } else if !character.is_whitespace()
            && !character.is_ascii_punctuation()
            && !matches!(
                character,
                '，' | '。' | '、' | '；' | '：' | '！' | '？' | '（' | '）'
            )
        {
            symbols += 1;
        }
    }

    let latin_word_bonus = text
        .split_whitespace()
        .map(|word| word.chars().filter(char::is_ascii_alphabetic).count())
        .filter(|length| *length >= 3)
        .sum::<usize>() as i64;

    if latin > cjk.saturating_mul(4) {
        latin * 3 + latin_word_bonus * 2 + digits - cjk * 10 - symbols * 6
    } else if cjk > latin {
        cjk * 5 + latin + other_letters * 2 + digits - symbols * 6
    } else {
        latin * 2 + cjk * 3 + other_letters * 3 + latin_word_bonus + digits - symbols * 6
    }
}

fn pdf_text_needs_ocr(text: &str) -> bool {
    if text.trim().is_empty() {
        return true;
    }
    let total = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count();
    if total == 0 {
        return true;
    }
    let suspicious = text
        .chars()
        .filter(|character| {
            matches!(*character, '\u{fffd}' | '\0')
                || character.is_control()
                || matches!(*character as u32, 0xe000..=0xf8ff)
        })
        .count();
    suspicious.saturating_mul(20) > total
        || text.contains("(cid:")
        || text.contains("锟斤拷")
        || text.contains("����")
}

fn pdf_text_quality(text: &str) -> i64 {
    let meaningful = text
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count() as i64;
    let suspicious = text
        .chars()
        .filter(|character| {
            matches!(*character, '\u{fffd}' | '\0')
                || character.is_control()
                || matches!(*character as u32, 0xe000..=0xf8ff)
        })
        .count() as i64;
    meaningful - suspicious * 20
}

#[cfg(not(windows))]
fn extract_pdf_ocr(_path: &Path) -> Result<String> {
    Err(anyhow!("图片型 PDF OCR 目前仅支持 Windows"))
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
    if matches!(kind, OoxmlKind::Excel) {
        return extract_xlsx(&mut archive);
    }
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
            OoxmlKind::Excel => false,
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

fn extract_xlsx(archive: &mut ZipArchive<File>) -> Result<String> {
    let shared_strings = if let Ok(mut entry) = archive.by_name("xl/sharedStrings.xml") {
        let mut xml = String::new();
        entry.read_to_string(&mut xml)?;
        xlsx_shared_strings(&xml)?
    } else {
        Vec::new()
    };
    let mut sheet_names = (0..archive.len())
        .filter_map(|index| {
            archive
                .by_index(index)
                .ok()
                .map(|entry| entry.name().to_owned())
        })
        .filter(|name| name.starts_with("xl/worksheets/sheet") && name.ends_with(".xml"))
        .collect::<Vec<_>>();
    sheet_names.sort_by_key(|name| natural_number(name));

    let mut output = String::new();
    for (sheet_index, name) in sheet_names.iter().enumerate() {
        let mut entry = archive.by_name(name)?;
        let mut xml = String::new();
        entry.read_to_string(&mut xml)?;
        let rows = xlsx_rows(&xml, &shared_strings)?;
        let headers = rows
            .iter()
            .find(|(_, cells)| xlsx_header_candidate(cells))
            .map(|(_, cells)| {
                cells
                    .iter()
                    .cloned()
                    .collect::<std::collections::HashMap<_, _>>()
            })
            .unwrap_or_default();
        for (row_number, cells) in rows {
            if cells.is_empty() {
                continue;
            }
            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(&format!("工作表 {} 第 {} 行", sheet_index + 1, row_number));
            for (column, value) in cells {
                let label = headers
                    .get(&column)
                    .filter(|header| !header.is_empty() && *header != &value)
                    .cloned()
                    .unwrap_or_else(|| xlsx_column_name(column));
                output.push_str(" | ");
                output.push_str(&label);
                output.push_str(": ");
                output.push_str(&value);
            }
        }
    }
    Ok(output)
}

fn xlsx_shared_strings(xml: &str) -> Result<Vec<String>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut values = Vec::new();
    let mut current = String::new();
    let mut in_item = false;
    let mut in_text = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(start)) => match start.local_name().as_ref() {
                b"si" => {
                    current.clear();
                    in_item = true;
                }
                b"t" if in_item => in_text = true,
                _ => {}
            },
            Ok(Event::Text(text)) if in_text => current.push_str(&text.unescape()?),
            Ok(Event::End(end)) => match end.local_name().as_ref() {
                b"t" => in_text = false,
                b"si" => {
                    values.push(normalize_text(&current));
                    in_item = false;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(error) => return Err(error.into()),
            _ => {}
        }
    }
    Ok(values)
}

fn xlsx_rows(xml: &str, shared_strings: &[String]) -> Result<Vec<(usize, Vec<(usize, String)>)>> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut rows = Vec::new();
    let mut row_number = 0usize;
    let mut cells = Vec::new();
    let mut cell_column = 0usize;
    let mut cell_type = String::new();
    let mut cell_value = String::new();
    let mut in_value = false;
    let mut in_inline_text = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(start)) => match start.local_name().as_ref() {
                b"row" => {
                    row_number = xml_attribute(&start, b"r")
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(rows.len() + 1);
                    cells.clear();
                }
                b"c" => {
                    cell_type = xml_attribute(&start, b"t").unwrap_or_default();
                    cell_column = xml_attribute(&start, b"r")
                        .map(|reference| xlsx_column_index(&reference))
                        .unwrap_or(cells.len());
                    cell_value.clear();
                }
                b"v" => in_value = true,
                b"t" if cell_type == "inlineStr" => in_inline_text = true,
                _ => {}
            },
            Ok(Event::Text(text)) if in_value || in_inline_text => {
                cell_value.push_str(&text.unescape()?)
            }
            Ok(Event::End(end)) => match end.local_name().as_ref() {
                b"v" => in_value = false,
                b"t" => in_inline_text = false,
                b"c" => {
                    let value = match cell_type.as_str() {
                        "s" => cell_value
                            .trim()
                            .parse::<usize>()
                            .ok()
                            .and_then(|index| shared_strings.get(index))
                            .cloned()
                            .unwrap_or_default(),
                        "b" => match cell_value.trim() {
                            "1" => "是".to_owned(),
                            "0" => "否".to_owned(),
                            value => value.to_owned(),
                        },
                        _ => normalize_text(&cell_value),
                    };
                    if !value.is_empty() {
                        cells.push((cell_column, value));
                    }
                }
                b"row" => rows.push((row_number, std::mem::take(&mut cells))),
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(error) => return Err(error.into()),
            _ => {}
        }
    }
    Ok(rows)
}

fn xml_attribute(start: &quick_xml::events::BytesStart<'_>, key: &[u8]) -> Option<String> {
    start
        .attributes()
        .flatten()
        .find(|attribute| attribute.key.as_ref() == key)
        .map(|attribute| String::from_utf8_lossy(attribute.value.as_ref()).into_owned())
}

fn xlsx_column_index(reference: &str) -> usize {
    reference
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .fold(0usize, |value, character| {
            value * 26 + (character.to_ascii_uppercase() as usize - 'A' as usize + 1)
        })
        .saturating_sub(1)
}

fn xlsx_column_name(mut index: usize) -> String {
    let mut value = String::new();
    loop {
        value.insert(0, (b'A' + (index % 26) as u8) as char);
        if index < 26 {
            break;
        }
        index = index / 26 - 1;
    }
    value
}

fn xlsx_header_candidate(cells: &[(usize, String)]) -> bool {
    cells.len() >= 2
        && cells.len() <= 64
        && cells.iter().all(|(_, value)| {
            let length = value.chars().count();
            (1..=40).contains(&length) && value.chars().any(|character| character.is_alphabetic())
        })
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
    fn xlsx_rows_preserve_headers_and_cell_relationships() {
        let shared = xlsx_shared_strings(
            r#"<sst><si><t>电影名</t></si><si><t>类型</t></si><si><t>适合人群</t></si>
            <si><t>欢乐家庭</t></si><si><t>喜剧 动画</t></si><si><t>全家观看</t></si></sst>"#,
        )
        .unwrap();
        let rows = xlsx_rows(
            r#"<worksheet><sheetData>
            <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
            <row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2" t="s"><v>4</v></c><c r="C2" t="s"><v>5</v></c></row>
            </sheetData></worksheet>"#,
            &shared,
        )
        .unwrap();
        let headers = rows[0]
            .1
            .iter()
            .cloned()
            .collect::<std::collections::HashMap<_, _>>();
        let rendered = rows[1]
            .1
            .iter()
            .map(|(column, value)| format!("{}: {value}", headers[column]))
            .collect::<Vec<_>>()
            .join(" | ");
        assert_eq!(
            rendered,
            "电影名: 欢乐家庭 | 类型: 喜剧 动画 | 适合人群: 全家观看"
        );
    }

    #[test]
    fn pdf_ocr_detection_is_conservative() {
        assert!(pdf_text_needs_ocr(""));
        assert!(pdf_text_needs_ocr("(cid:123) (cid:456)"));
        assert!(!pdf_text_needs_ocr(
            "供应商逾期付款时，应当按照合同约定支付违约金。"
        ));
    }

    #[cfg(windows)]
    #[test]
    fn ocr_scoring_prefers_clean_latin_text() {
        let clean = "Mobile support is available on Android and iOS devices.";
        let noisy = "Mobile SUPP0rt is available 0 n Andr01d 和 iO§ devices.";
        assert!(ocr_text_score(clean) > ocr_text_score(noisy));
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
