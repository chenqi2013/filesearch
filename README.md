# 本地知搜 MVP

面向 Windows 10/11 的本地文档搜索桌面应用。项目采用 Tauri 2 + React + TypeScript 桌面界面、Rust 外壳，以及可独立运行的 Rust 搜索核心服务。

## 已实现

- 添加、移除一个或多个本地索引目录
- PDF、DOCX、XLSX、PPTX、TXT、MD、CSV、JSON、LOG、RST 文本抽取
- SQLite WAL 文件元数据 + Tantivy BM25 分段倒排索引，可面向 5,000–50,000 个文件
- EmbeddingRWKV Tiny 本地 ONNX Embedding，支持关键词、语义、混合三种检索模式
- 按文件大小和修改时间增量重建，文件变化 900 ms 去抖后后台自动更新
- 文件类型过滤、命中片段、相关度、大小和修改时间
- 打开原文件、在资源管理器中定位文件
- 索引进度、文档/片段计数、失败文件列表
- 点击文档、片段、失败统计可分页查看明细；文档和片段可直接打开源文件
- 中文、英文、俄文界面
- 索引仅保存在本机应用数据目录，不上传文档

## 快速运行

环境要求：Node.js 20+、pnpm 10+、Rust 1.87+。Windows 需要 WebView2 Runtime 和 Visual Studio C++ Build Tools。

```bash
pnpm install
pnpm desktop:dev
```

`desktop:dev` 会先编译独立搜索服务为 Tauri sidecar，再启动桌面应用。

只运行搜索核心：

```bash
cargo run -p search-core -- --listen 127.0.0.1:47653 --data-dir .filesearch-data
```

只运行 Web 界面：

```bash
pnpm dev
```

Web 开发界面位于 [http://127.0.0.1:1420](http://127.0.0.1:1420)。浏览器模式不能选择或打开本地文件，需要同时启动搜索核心；完整功能请使用 Tauri。

## 构建 Windows 安装包

在 Windows 构建机运行：

```powershell
pnpm install
pnpm desktop:build
```

安装包输出到 `target/release/bundle/`。`scripts/prepare-sidecar.mjs` 会根据当前 Rust target triple 生成 Tauri 所需的 sidecar 文件名。

## 数据位置

桌面版把 SQLite、Tantivy 和本地模型缓存放在 Tauri 的应用本地数据目录：

- Windows：`%LOCALAPPDATA%\com.localfind.desktop\`
- macOS：`~/Library/Application Support/com.localfind.desktop/`

其中 `search.db` 保存文件元数据、失败记录和文档级向量，`tantivy/` 保存全文倒排索引。EmbeddingRWKV Tiny 文本 ONNX 模型随 Windows 安装包内置，应用只读取源文件，不会修改、移动或删除源文件。

语义索引使用内置的 `EmbeddingRWKV Tiny` 模型（768 维），采用官方 World Tokenizer、EOS 结尾标记和 `[RETR]` 检索头；每个文档只对文件名及最多约 4,000 字符正文编码一次，正文分块仍由 Tantivy 提供关键词定位。模型和推理均在本机完成，文档内容不会上传，也不会从网络下载模型。模型文件缺失或推理失败时，会自动使用离线中文词项向量，关键词检索和索引服务仍可用。

完全离线环境可在启动前设置 `FILESEARCH_EMBEDDING_OFFLINE=1`，直接使用离线特征。

## 大数据量设计

- SQLite 使用 WAL、批量 4 条 Embedding 和事务写入，索引过程内存有界。
- Tantivy 只保存分段词项与定位 ID，正文和状态由 SQLite 管理，避免 JSON 全量读写。
- 语义层保存每个文档一个 768 维向量；5 万文档约 146 MB 原始向量数据，查询时顺序扫描并与 Tantivy 候选融合。
- 文件监听自动合并短时间内的批量变化；共享盘断线时保留已有结果，恢复后自动补扫。
- 关键词查询跳过向量读取，单文件增量事件只读取轻量路径映射，避免在 5 万文件规模下反复载入约 146 MB 向量。

运行 50,000 文档 SQLite 容量测试：

```bash
cargo test -p search-core sqlite_wal_handles_fifty_thousand_documents -- --ignored
```

## 当前边界

- `.doc`、`.xls`、`.ppt` 旧版二进制格式尚未解析；可先另存为 OOXML 格式。
- 扫描版 PDF 与图片 OCR、问答摘要尚未实现；加密 PDF 会明确列为失败文件。
- 已完成 5 万条合成元数据容量测试；上线前仍建议在目标 Windows 机器和真实混合文档/共享盘环境进行长时间压力测试。

详细设计见 [docs/architecture.md](docs/architecture.md)，测试记录见 [docs/test-report.md](docs/test-report.md)。
