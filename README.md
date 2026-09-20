# AI 学习路线与作品集

26 周(6 个月,每周 6h)的 AI 工程学习记录。

> 背景:深度学习研究出身,现职后端开发。
> 定位:做"懂 AI 的工程师"——从应用切入,往基础设施扎根。

完整计划与每周任务清单见 [ai-learning-roadmap.md](ai-learning-roadmap.md)。

## 四个阶段

| 阶段 | 周数 | 目标 |
| --- | --- | --- |
| 认知重建 | 1–4 | 主流 LLM 概念有直觉,API 玩到熟 |
| 生态与选型 | 5–8 | 开源模型不再是黑话,知道什么时候该用什么 |
| 应用硬功夫 | 9–16 | 独立交付敢给真实用户用的 RAG 和 Agent |
| 综合项目 | 17–26 | 自建模型服务层:vLLM 部署、压测、网关、可观测 |

## 使用规则

1. 每周必须有产出(代码、笔记、demo 都算),纯看视频不超过当周时间的 1/3。
2. 卡住超过 45 分钟就换方式:问 AI、看源码、跳过,别死磕。
3. 每 4 周一次复盘周,允许落后,不允许断更。
4. 不追热点新闻,新模型发布与我无关,除非它进了我的项目。

## 目录

随学习进度更新:

- `ai-learning-roadmap.md` — 26 周完整计划
- 每周产出(笔记、脚本、demo)按周归档

### 第 1 周 · 文本建模与 Transformer

- `llm-fundamentals-notes.md` — 文本建模基础复习:bigram(统计与 NN 两条路)→ embedding → MLP(Bengio 2003),附训练实战坑与自测题
- `transformer-code-reading-guide.md` — Transformer 直觉重建 + nanoGPT 带行号精读指南 + 伪代码练习
- `nanogpt-model.py` — 精读用源码,取自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)(MIT License),仅作阅读参考
- `deep-dive-llm-pipeline-notes.md` — 预训练 → SFT → RLHF/RLVR 全流程笔记,重点 tokenizer / SFT 数据构造 / RLVR 演进
