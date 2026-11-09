import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, ExternalLink, X } from "lucide-react";
import { coreApi } from "./api";
import type { Locale } from "./i18n";
import type { DocumentContent, SearchFilters, SearchResult } from "./types";

export const featureMessages = (locale: Locale) => locale === "zh" ? {
  filters: "更多筛选", directory: "全部目录", scope: "搜索范围", all: "文件名和正文", name: "仅文件名", content: "正文",
  after: "修改时间从", before: "修改时间至", min: "最小 MB", max: "最大 MB", phrase: "必须包含完整短语", exclude: "排除词（用逗号分隔）", apply: "应用筛选", reset: "重置",
  preview: "内容预览", related: "相关片段", jump: "定位到此处", source: "选择原文位置", missing: "当前索引没有页码信息；重新解析该文件后可补充。", previous: "上一页", next: "下一页", close: "关闭", open: "打开原文件", chunk: "片段", loading: "正在加载…",
  retry: "重试", reparse: "重新解析此文件", retryAll: "重试全部失败文件", retryPage: "重试本页失败文件", skipped: "已跳过", everyday: "日常使用（让出 CPU）", full: "全速索引", resources: "索引资源模式", pending: "待完成语义", textReady: "文本已就绪", diagnostics: "诊断与日志", export: "导出诊断", relevance: "排序分数", scoreHint: "仅用于同一次搜索的排序，不代表准确率或置信度。", candidateHint: "显示召回候选范围内的结果，目录和其他筛选可帮助缩小范围。", previewHint: "展示提取的文本；排版以原文件为准。", page: "页",
} : locale === "ru" ? {
  filters: "Фильтры", directory: "Все папки", scope: "Область", all: "Имя и текст", name: "Только имя", content: "Текст", after: "Изменён с", before: "Изменён до", min: "Мин. МБ", max: "Макс. МБ", phrase: "Точная фраза", exclude: "Исключить (через запятую)", apply: "Применить", reset: "Сбросить", preview: "Текст документа", related: "Похожие фрагменты", jump: "Перейти", source: "Позиция в оригинале", missing: "Нет номеров страниц. Повторная обработка добавит их.", previous: "Назад", next: "Далее", close: "Закрыть", open: "Открыть оригинал", chunk: "Фрагмент", loading: "Загрузка…", retry: "Повторить", reparse: "Обработать заново", retryAll: "Повторить все ошибки", retryPage: "Повторить эту страницу", skipped: "Пропущено", everyday: "Повседневный режим", full: "Полная скорость", resources: "Ресурсы индексации", pending: "Ожидает векторов", textReady: "Текст готов", diagnostics: "Диагностика", export: "Экспорт диагностики", relevance: "Баллы ранжирования", scoreHint: "Баллы сортировки в этом поиске, не вероятность правильного ответа.", candidateHint: "Показаны найденные кандидаты. Фильтры уточняют поиск.", previewHint: "Извлечённый текст; оформление см. в оригинале.", page: "Стр.",
} : {
  filters: "More filters", directory: "All folders", scope: "Search scope", all: "Name and content", name: "Filename only", content: "Content", after: "Modified from", before: "Modified through", min: "Minimum MB", max: "Maximum MB", phrase: "Required exact phrase", exclude: "Exclude terms (comma separated)", apply: "Apply filters", reset: "Reset", preview: "Content preview", related: "Related passages", jump: "Go to passage", source: "Source location", missing: "Page information is unavailable. Reparse this file to add it.", previous: "Previous", next: "Next", close: "Close", open: "Open original", chunk: "Passage", loading: "Loading…", retry: "Retry", reparse: "Reparse this file", retryAll: "Retry all failed files", retryPage: "Retry failures on this page", skipped: "Skipped", everyday: "Everyday use (yield CPU)", full: "Full speed", resources: "Index resource mode", pending: "Semantic pending", textReady: "Text ready", diagnostics: "Diagnostics and logs", export: "Export diagnostics", relevance: "Ranking score", scoreHint: "For ordering this search only, not an accuracy or confidence percentage.", candidateHint: "Results within the retrieval candidate pool. Use filters to narrow the search.", previewHint: "Extracted text; see the original for formatting.", page: "Page",
};

export function SearchFilterPanel({ locale, directories, onApply }: { locale: Locale; directories: string[]; onApply: (filters: SearchFilters) => void }) {
  const f = featureMessages(locale);
  const [draft, setDraft] = useState({ directory: "", scope: "all", after: "", before: "", min: "", max: "", phrase: "", exclude: "" });
  const [error, setError] = useState("");
  const set = (key: keyof typeof draft, value: string) => setDraft(previous => ({ ...previous, [key]: value }));
  return <details className="search-filter-panel"><summary>{f.filters}</summary><div className="search-filter-fields">
    <label>{f.directory}<select value={draft.directory} onChange={e => set("directory", e.target.value)}><option value="">{f.directory}</option>{directories.map(path => <option key={path} value={path}>{path}</option>)}</select></label>
    <label>{f.scope}<select value={draft.scope} onChange={e => set("scope", e.target.value)}><option value="all">{f.all}</option><option value="name">{f.name}</option><option value="content">{f.content}</option></select></label>
    <label>{f.after}<input type="date" value={draft.after} onChange={e => set("after", e.target.value)} /></label>
    <label>{f.before}<input type="date" value={draft.before} onChange={e => set("before", e.target.value)} /></label>
    <label>{f.min}<input type="number" min="0" step="any" value={draft.min} onChange={e => set("min", e.target.value)} /></label>
    <label>{f.max}<input type="number" min="0" step="any" value={draft.max} onChange={e => set("max", e.target.value)} /></label>
    <label>{f.phrase}<input value={draft.phrase} onChange={e => set("phrase", e.target.value)} /></label>
    <label>{f.exclude}<input value={draft.exclude} onChange={e => set("exclude", e.target.value)} /></label>
    <div className="filter-actions"><button className="primary-button" onClick={() => {
      if ([draft.min, draft.max].some(value => value && (!Number.isFinite(Number(value)) || Number(value) < 0)) || (draft.min && draft.max && Number(draft.min) > Number(draft.max)) || (draft.after && draft.before && draft.after > draft.before)) { setError(locale === "zh" ? "请输入有效范围（大小不能为负数）" : "Invalid range"); return; }
      setError("");
      onApply({ directory: draft.directory || undefined, scope: draft.scope as SearchFilters["scope"], modified_after: draft.after ? new Date(`${draft.after}T00:00:00`).getTime() : undefined, modified_before: draft.before ? new Date(`${draft.before}T23:59:59.999`).getTime() : undefined, min_size: draft.min ? Math.round(Number(draft.min) * 1024 ** 2) : undefined, max_size: draft.max ? Math.round(Number(draft.max) * 1024 ** 2) : undefined, exact_phrase: draft.phrase.trim() || undefined, exclude_terms: draft.exclude.split(/[,，]/).map(s => s.trim()).filter(Boolean) });
    }}>{f.apply}</button><button className="secondary-button" onClick={() => { setDraft({ directory: "", scope: "all", after: "", before: "", min: "", max: "", phrase: "", exclude: "" }); setError(""); onApply({}); }}>{f.reset}</button>{error && <span role="alert">{error}</span>}</div>
  </div></details>;
}

export function DocumentPreview({ result, query, locale, onClose, onOpen }: { result: SearchResult; query: string; locale: Locale; onClose: () => void; onOpen: (path: string) => void }) {
  const f = featureMessages(locale);
  const [data, setData] = useState<DocumentContent | null>(null);
  const [offset, setOffset] = useState(0);
  const [target, setTarget] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [locationQuery, setLocationQuery] = useState("");
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let active = true;
    setLoading(true); setError("");
    coreApi.content(result.id, offset, query).then(value => { if (active) setData(value); }).catch(error => { if (active) setError(String(error)); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [result.id, offset, query]);
  useEffect(() => { if (data && !loading && target !== null) container.current?.querySelector(`[data-position="${target}"]`)?.scrollIntoView({ block: "center" }); }, [data, loading, target]);
  const jump = (position: number) => { setTarget(position); setOffset(Math.floor(position / 30) * 30); };
  const locations = data?.locations.filter(location => location.label.includes(locationQuery)).slice(0, 300) ?? [];
  return <div className="modal-backdrop" onMouseDown={onClose}><section className="modal preview-modal" role="dialog" aria-modal="true" aria-label={f.preview} onMouseDown={e => e.stopPropagation()}>
    <header><h2>{f.preview} · {result.name}</h2><button className="icon-button" onClick={onClose} title={f.close}><X /></button></header>
    <div className="preview-toolbar"><span>{f.previewHint}</span><button className="secondary-button" onClick={() => onOpen(result.path)}><ExternalLink />{f.open}</button></div>
    {error && <p role="alert">{error}</p>}
    {data && <>
      {data.locations.length > 300 && <label className="preview-location-filter">{locale === "zh" ? "按页码或行号筛选位置（下拉框最多显示 300 项）" : "Filter locations by page or row (up to 300 options)"}<input value={locationQuery} onChange={e => setLocationQuery(e.target.value)} /></label>}
      {data.locations.length > 0 ? <select aria-label={f.source} value="" onChange={e => jump(Number(e.target.value))}><option value="">{f.source}</option>{locations.map((location, index) => <option key={index} value={Math.floor(location.start / 700)}>{location.label}</option>)}</select> : result.extension === "pdf" && <p className="muted">{f.missing}</p>}
      <div className="preview-body" ref={container} aria-busy={loading}>
        {offset === 0 && data.passages.length > 0 && <section className="preview-related"><h3>{f.related}</h3>{data.passages.map((passage, index) => <article key={index}><p><Match text={passage.text} query={query} /></p><button className="secondary-button" onClick={() => jump(passage.position)}>{f.jump} · #{passage.position + 1}</button></article>)}</section>}
        {data.chunks.items.map(chunk => {
          const start = chunk.position * 700;
          const locations = data.locations.filter(location => location.start < start + Array.from(chunk.text).length && location.end > start);
          const labels = locations.length > 4 ? `${locations[0].label} … ${locations[locations.length - 1].label}` : locations.map(location => location.label).join(" / ");
          return <article className={`preview-chunk ${target === chunk.position ? "active" : ""}`} key={chunk.id} data-position={chunk.position}><small>{f.chunk} #{chunk.position + 1}{labels && ` · ${labels}`}</small><p><Match text={chunk.text} query={query} /></p></article>;
        })}
      </div>
      <footer><span>{loading ? f.loading : `${offset + 1}–${Math.min(offset + data.chunks.items.length, data.chunks.total)} / ${data.chunks.total}`}</span><div className="pagination"><button className="icon-button" disabled={loading || offset === 0} onClick={() => setOffset(Math.max(0, offset - 30))} title={f.previous}><ChevronLeft /></button><select aria-label={f.page} value={Math.floor(offset / 30)} disabled={loading} onChange={e => setOffset(Number(e.target.value) * 30)}>{Array.from({ length: Math.max(1, Math.ceil(data.chunks.total / 30)) }, (_, i) => <option key={i} value={i}>{f.page} {i + 1}</option>)}</select><button className="icon-button" disabled={loading || offset + 30 >= data.chunks.total} onClick={() => setOffset(offset + 30)} title={f.next}><ChevronRight /></button></div></footer>
    </>}
    {!data && !error && <p>{f.loading}</p>}
  </section></div>;
}

function Match({ text, query }: { text: string; query: string }) {
  const index = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
  return index < 0 ? <>{text}</> : <>{text.slice(0, index)}<mark>{text.slice(index, index + query.length)}</mark>{text.slice(index + query.length)}</>;
}

export function downloadJson(name: string, value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json;charset=utf-8" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name; anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
