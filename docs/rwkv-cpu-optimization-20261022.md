# RWKV CPU Embedding 优化实测（2026-10-22）

## 本轮交付

本地模型资产与搜索核心源码采用新的混合 INT8 配置；release 搜索核心已重新编译。仍使用同一个 EmbeddingRWKV Tiny checkpoint、768 维输出、官方 World 词表和单个 EOS，没有切换 E5/Qwen。单条上限仍是 1,024 Token（含 EOS）；全文继续按 480 字符、400 步长切语义窗口，没有限制窗口数量或通过抽样加速。

仅扩大部分矩阵的量化范围：

| 配置 | FFN 动态 INT8 | 注意力输出投影动态 INT8 | 其他路径 |
| --- | --- | --- | --- |
| v9 原版 | 第 6–11 层 | 无 | FP32 |
| v10 本地优化版 | 第 3–11 层 | 第 6–11 层 | FP32 |

层号从 0 开始；量化为 per-channel QInt8，未使用 reduce-range。生产批量仍最多 4 窗口、2,048 Token，且只合并相同 Token 长度；线程配置和 WKV 数学实现不变。

旧 ONNX 已备份到 `.codex-tmp/cpu-opt-20261022/production-v9.onnx`。新 ONNX 同步到 `assets/models/embedding-rwkv-tiny/model.onnx` 和 `target/release/models/embedding-rwkv-tiny/model.onnx`；未修改用户原始文档、运行中的安装版或其数据库，未启动桌面应用、未打包安装程序。

## 同机对照结果

设备是 i7-13700H、16 GB RAM，CPU 后端和项目自带 ONNX Runtime 1.22。存在背景负载及温度/调度波动，结果只能代表本机的具体样本。

小说原文件为 6,990,767 bytes，规范化全文 2,226,641 字符、5,567 个窗口。文件 SHA-256：`8f66af22bf03e0411f157e8173a5f0fabff5ca9d07c8de63bf590bea547e96d9`。

### 纯推理

从小说均匀选出同一组 96 窗口，含相同文件名前缀，共 45,370 Token；每次重启推理进程，不使用向量缓存，加载与预热不计入推理时间。执行顺序为旧→新→新→旧→旧→新。

| 配置 | 三次推理耗时 | 中位数 |
| --- | --- | --- |
| 原版 | 25.850 / 28.297 / 27.440 秒 | 27.440 秒 |
| 优化版 | 23.712 / 24.573 / 24.858 秒 | 24.573 秒 |

中位耗时减少约 **10.45%**，三对相邻实验都有正收益。小说窗口向量与原版余弦最小 0.998263、平均 0.998823，向量不是完全一致。

### 实际索引流程

从小说均匀抽取 64 段、每段 480 字符，构成相同的 83,734 bytes 测试副本，正式索引器实际生成 67 个语义窗口。每轮使用独立数据库与全新缓存，旧→新→新→旧交替；包含模型首次加载，期间验证暂停恢复。

| 配置 | 两轮 Embedding 阶段 | 平均 | 进程峰值内存（两次均值） |
| --- | --- | --- | --- |
| 原版 | 18.197 / 20.365 秒 | 19.281 秒 | 920.1 MiB |
| 优化版 | 16.622 / 17.845 秒 | 17.234 秒 | 831.7 MiB |

该样本 Embedding 阶段平均减少 **10.62%**；进程峰值内存平均减少约 88.4 MiB（92.7 MB，约 9.6%）。内存为 Windows 进程 peak working set，包含推理服务本身，不是全系统占用。

优化版两轮暂停响应约 0.448 / 0.453 秒；暂停时窗口数保持不变，恢复后完成。`/stats` 最大响应约 26.7 ms；每轮观察到 23 次窗口进度变化，无失败。

关闭并重启服务、仅修改测试副本 mtime 后，同内容重建的 Embedding 为 18 / 16 ms。这是持久缓存命中，**不能当作首次推理耗时**。

本轮没有再次跑完整 6.99 MB 小说，因此没有新版整本耗时结论。前次原版全文约 23 分 52 秒，也不能直接拿来宣称本次全文提速。首次全文语义依然随 Token 数增长，本轮约 10% 的样本收益不会让大文件变成秒级。

## 检索质量

| 测试 | 原版 | 优化版 |
| --- | --- | --- |
| 本地 20 文档、40 查询首位命中 | 39/40 | 39/40 |
| 本地 Recall@3 | 100% | 100% |
| SciFact 子集首位命中 | 37/50 | 37/50 |
| SciFact Recall@3 | 86.5% | 86.5% |
| SciFact Recall@10 | 96% | 96% |
| SciFact MRR@3 | 0.7967 | 0.7933 |

公开样本来自 NanoBEIR-en/SciFact：保留 50 条查询的全部相关文档，加 200 篇确定性干扰文档，共 255 文档，原始语料 2,919 文档。它是英文小子集，**不是完整 NanoBEIR/MTEB 评测**，不能代表所有中文客户数据。纯模型对照用单条推理以隔离量化影响；最终模型还通过实际合批/单条向量一致性测试（余弦均大于 0.999999）。

少数查询排名发生变化（例如一个相关文档从第 2 降到第 3），所以不能宣称准确率完全不变。核心召回指标在已测样本中持平，MRR@3 略降；仍需更多真实中文资料回归。客户合同、岗位、电影、歌单模型回归及 PDF 协议生产输入形状测试通过；这些用的是自建样例，并非客户原文件，扫描 PDF 解析问题也不由本次量化测试覆盖。

## 向量协议与升级

`embedding::EMBEDDING_PROFILE` 已从 v9 升为 `rwkv7-tiny-ffn3-attnout6-int8-structured-full-passages-v10`。量化变化后，旧正文向量不能与新查询向量混用；旧缓存键也不能命中新版。

下次启动新版搜索核心时，现有协议检测会清除旧语义向量，并对已配置目录自动重建。源文件与关键词索引保留；首次升级可能需要较长时间。新增测试验证 v9 必须重建且相同文本的旧缓存不能被 v10 复用；当前 v10 缓存重启后可正常命中。

最终 release 在隔离副本中执行暂停、强制结束和重启：11 个已完成窗口全部复用；最终生成全部窗口，无失败。`pipeline-v10-interrupted/result.json` 的 11.021 秒只计重启后的剩余 Embedding，不能与上面首次完整样本对照。

## 未采用的实验

- 全部 FFN 直接 INT8：96 窗口约快 17.6%，但本地首位命中 39→38，拒绝作为默认。
- WKV 两遍循环融合：结果一致，但约慢 6.6%，已撤回。
- WKV decay 向量 exp 近似：仅约快 1.45%，且产生数值差异，已撤回。
- 批量上限 8、4,096 Token：256 窗口两轮旧 73.837 / 56.580 秒，新 67.435 / 59.558 秒；第二轮反而变慢，没有稳定收益，默认仍为 4。
- 仅限制到本机 P 核：中位耗时反而约慢 6.5%，不采用核绑定。
- ORT 1.29、reduce-range 等组合没有足够稳定的收益，运行时及其他默认设置不变。

## 验证与复现

- 最终常规 Rust 测试：70 通过，0 失败，5 项显式慢/模型测试默认跳过。
- 另行运行 4 项模型测试全部通过，包括本地 40 查询检索、合批一致性和客户语义案例。
- `cargo build --release -p search-core --bin search-core --example rwkv-embed` 成功；仅有已有依赖/未使用代码警告。
- Rust 格式检查、`git diff --check`、五个相关 Python 脚本语法检查通过；仅清理本轮生成的 Python 字节码缓存，其他测试资料保留。
- 新增完整 1,024 Token 的 WKV 标量/AVX 输出一致性回归。

原始结果在 `.codex-tmp/cpu-opt-20261022/`：`ffn3-output6-comparison.json`、`ffn3-output6-scifact.json`、`pipeline-0` 到 `pipeline-3/result.json`、`pipeline-v10-interrupted/result.json`。

关键资产 SHA-256：

| 资产 | SHA-256 |
| --- | --- |
| v9 ONNX | `f5a0701fc1263cec1b0d46c11031abec5142697b2da21621bf362cf217fdea7d` |
| v10 ONNX | `b2cfe70957dcabd96c85ab8d4c4928060fd060d74420b562b30889a8f818bead` |
| 内置 ORT DLL | `579b636403983254346a5c1d80bd28f1519cd1e284cd204f8d4ff41f8d711559` |
| 冻结基准 encoder EXE | `cfb271c044510c51b8e8d6c98481c8949aa62a1b36c89a58319456c9c21bec95` |
| 最终 release search-core EXE | `c1868eb800f24217fd1569deef88f98dd8561f90cb213a2fd3a4cca462218f2a` |

量化复现（输入必须是原始 FP32 导出图）：

```powershell
python -B scripts/quantize-embedding-rwkv.py --input .codex-tmp/rwkv-corrected-export/model.onnx --output <新输出路径>
# 旧配置：追加 --ffn-start 6 --attention-output-start 12

# 独立索引/暂停/中断恢复，不接触用户数据库；输出目录必须全新
python -B -u scripts/verify-long-txt-pipeline.py --file '<小说路径>' --output .codex-tmp/v10-recheck --interrupt-after-windows 8
# --full 才会对整本小说测试，默认使用 83,734 bytes 副本
```

同机纯推理对照：

```powershell
python -B -u scripts/compare-rwkv-cpu.py --file '<小说路径>' --baseline .codex-tmp/cpu-opt-20261022/production-v9.onnx --candidate assets/models/embedding-rwkv-tiny/model.onnx --baseline-encoder .codex-tmp/cpu-opt-20261022/rwkv-embed-before.exe --candidate-encoder .codex-tmp/cpu-opt-20261022/rwkv-embed-before.exe --dataset .codex-tmp/semantic-retrieval-eval.json --samples 96 --repeats 3 --output .codex-tmp/v10-compare.json
```

对照工具记录模型、运行时、encoder 与语料哈希；`--samples 0` 才是全文纯推理，`--retrieval-only` 单独验证检索。公开子集可用 `scripts/build-rwkv-public-fixture.py` 从 NanoBEIR parquet 重建。模型 ONNX 是现有忽略资产，提交源码本身不会包含新模型，分发时需使用已同步的新资产。
