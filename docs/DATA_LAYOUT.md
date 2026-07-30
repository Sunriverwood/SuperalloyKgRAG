# 数据目录布局 (Data Layout)

> 更新时间：2026-07-30  
> 说明：本库 `data/` 与实验归档目录按「主流程产物 + 按实验分子目录」组织，与评测/消融脚本约定一致。

## 顶层概览

```
SuperalloyKgRAG/
├── data/                 # 主数据（gitignore，需本地保留）
├── history/              # 历史消融/调试归档（gitignore，非主流程）
├── research_paper/       # 组会/汇报材料（gitignore，非代码数据）
├── visualizations/       # 论文图与分析中间结果（入库）
└── logs/                 # 运行日志
```

## `data/` 主结构

```
data/
├── original_data/              # 原始语料
│   ├── books/                  # 书籍 PDF（切片）
│   ├── full_text/              # 论文全文 PDF
│   └── abstract/               # 摘要 Excel
├── processed_jsons/            # VLM/论文解析后的 JSON
├── chunks/                     # 文本单元
│   ├── text_units.jsonl
│   ├── abstract_units.jsonl
│   ├── table_units.jsonl
│   └── image_units.jsonl
├── graphs/                     # 知识图谱
│   ├── extracted/              # 抽取阶段
│   ├── enriched/               # 富化阶段（含 text_only 变体）
│   ├── final_graph.json        # 主图（默认配置）
│   ├── merged_graph.json
│   ├── disambiguation_graph.json
│   ├── final_graph.gexf
│   ├── text_only/              # 消融：仅文本建图
│   └── no_entities_merge/      # 消融：不做实体合并
├── embeddings/                 # LanceDB
│   ├── enriched.db             # 主库
│   ├── basic_rag_multimodal.db
│   ├── basic_rag_text.db
│   ├── enriched_text_only.db
│   └── no_entities_merge.db
├── reports/                    # 社区报告与评测报告
│   ├── community_summaries.jsonl
│   ├── text_only/
│   ├── no_entities_merge/
│   ├── multidimensional_evaluation/
│   ├── rescore/                # 按实验子目录镜像 answers
│   └── analysis/               # 分层得分 Excel 等
├── reasoning/                  # 图推理权重
│   ├── develop.pt              # 主模型
│   ├── text_only.pt
│   └── no_entities_merge.pt
├── cache/                      # 批请求/中间缓存（含消融子目录）
├── evaluation_sets/            # 评测题
│   ├── L12.json                # L1+L2
│   ├── L3.json
│   ├── L4.json
│   └── hard.json               # 教材深推理子集
└── answers/
    └── multidimensional_evaluation/
        ├── old-baseline/       # 早期 baseline + RAG 完整答案
        ├── new-baseline/       # 更新后的 baseline 对比
        ├── ablation_*/         # 各消融实验答案
        ├── baisc_rag_textonly/ # 历史命名保留
        ├── unstrict/           # 非严格设定下的消融
        └── error/              # 失败/调试运行留存
```

## 实验子目录约定

| 用途 | 答案目录 | 重打分输出 | 说明 |
|------|----------|------------|------|
| 主多维评测命名运行 | `answers/.../<run_dir>/` | `reports/rescore/<run_dir>/` | `--run-dir new-baseline` |
| 消融 | `answers/.../ablation_<name>/` | `reports/rescore/ablation_<name>/` | `--ablation text_only` |
| 分层分析 | — | `reports/analysis/` | `rescore_level_analysis --dir ...` |

常用命令：

```bash
# 写入命名实验目录
python -m evaluation.multidimensional_evaluator --run-dir new-baseline

# 消融（自动写入 ablation_<name>/）
python -m evaluation.multidimensional_evaluator --ablation text_only --methods local,reasoning

# 对某实验子目录重打分
python -m evaluation.rescore --answers_dir new-baseline
python -m evaluation.rescore --answers_dir ablation_text_only

# 分层得分导出
python -m evaluation.rescore_level_analysis --dir new-baseline
python -m evaluation.rescore_level_analysis --all
```

路径细节以 `config/settings.yaml` 中 `ablation:` 与 `multidimensional_evaluation:` 为准。

## `history/`（可选归档）

| 子目录 | 内容 |
|--------|------|
| `graph_builder_history_versions/` | 旧版建图脚本（代码侧亦见 `core/pipeline/graph_builder_history_versions/`） |
| `no_entities_merge_no_communities/` | 「无合并 + 无社区」旁路实验大图/向量库 |
| `discrete/` | 离散评分实验快照 |
| `test/` | 测试材料 |

非默认配置路径，主流程不读取。

## `research_paper/`

组会 PPT 与幻灯片导出，与运行时数据无关。
