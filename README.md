# 本地知搜 MVP

面向 Windows 10/11 的本地文档搜索桌面应用。项目采用 Tauri 2 + React + TypeScript 桌面界面、Rust 外壳，以及可独立运行的 Rust 搜索核心服务。

## 已实现

- 添加、移除一个或多个本地索引目录
- PDF、DOCX、XLSX、PPTX、TXT、MD、CSV、JSON、LOG、RST 文本抽取
- 按文件大小和修改时间执行增量重建
- 关键词、轻量语义、混合三种检索模式
- 文件类型过滤、命中片段、相关度、大小和修改时间
- 打开原文件、在资源管理器中定位文件
- 索引进度、文档/片段计数、失败文件列表
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

桌面版把 `index.json` 存储在 Tauri 的应用本地数据目录：

- Windows：`%LOCALAPPDATA%\com.localfind.desktop\index.json`
- macOS：`~/Library/Application Support/com.localfind.desktop/index.json`

删除该文件即可清理索引。应用只读取源文件，不会修改、移动或删除源文件。

## MVP 边界

- “语义搜索”使用本地中文字符/词项特征哈希向量，零模型下载、零云请求，适合验证自然语言检索流程，但同义词理解不及神经网络 Embedding。
- `.doc`、`.xls`、`.ppt` 旧版二进制格式尚未解析；可先另存为 OOXML 格式。
- 扫描版 PDF 与图片 OCR、问答摘要、文件系统实时监听不在本版范围内。
- 当前持久化格式为 JSON，适合 MVP 与约数千份文档验证；生产版 5 万文件规模建议替换为 SQLite/Tantivy 与本地向量索引。

详细设计见 [docs/architecture.md](docs/architecture.md)，测试记录见 [docs/test-report.md](docs/test-report.md)。
