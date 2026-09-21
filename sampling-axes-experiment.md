# 三轴控制实验:采样随机性 × 推理强度 × 输出约束

> 第 2 周动手任务。替代原「温度 vs top_p 对比」。
> 目标不是记住参数含义,而是**产出你自己项目用的参数策略表**。

---

## 0. 为什么是"三轴"

温度/top_p 只是其中一根旋钮。把它们放在正确的坐标系里,才不会问出"现在还用得到温度吗"这种问题——因为推理强度**不替代**采样控制,两者正交。

| 轴 | 控制什么 | 参数 | 影响 |
|---|---|---|---|
| **① 采样随机性** | 输出的随机程度 | temperature、top_p、top_k | 多样性 / 可复现性 |
| **② 推理强度** | 思考的深度与计算量 | reasoning effort(low/medium/high) | 准确率 / 延迟 / 成本 |
| **③ 输出约束** | 输出的形状 | JSON schema、stop、max_tokens | 可解析性 |

三条组合起来才是完整的"模型行为控制面板"。

---

## 1. 实验要回答的三个问题

1. **哪些任务必须 T=0?**(否则评测不可复现)
2. **哪些任务提升推理强度真的有用,哪些是浪费钱?**
3. **结构化输出约束能替代 prompt 里"请返回 JSON"吗?成功率高多少?**

---

## 2. 任务集设计(三类,每类 5–10 条)

| 类型 | 例子 | 期望答案 |
|---|---|---|
| **A. 确定性任务** | 工单分类、字段抽取 | 唯一正确答案,可自动判定 |
| **B. 生成任务** | 客服草稿、文案改写 | 有质量高低,但无唯一答案 |
| **C. 推理任务** | 多步数学题、逻辑题 | 唯一正确答案,需要推导 |

**每条任务准备两个东西**:输入 + 判定标准。A/C 类用精确匹配或规则判定;B 类用评分 rubric 或人工打分。

---

## 3. 配置矩阵:不要做笛卡尔积

**错误做法**:4 档温度 × 3 档强度 × 2 种格式 × 3 类任务 × 10 次重复 = 720 次调用。又贵又慢,而且大部分组合没有信息量。

**正确做法:分阶段正交设计**(One-Factor-At-A-Time + 定点交叉)。

### 阶段一:单轴扫描(固定其他轴为默认值)

| 扫描轴 | 取值 | 固定其他轴 | 调用数 |
|---|---|---|---|
| 温度 | 0 / 0.3 / 0.7 / 1.0 | effort=medium, 自由文本 | 4 × 3 类 × 5 条 = 60 |
| 推理强度 | low / medium / high | T=0, 自由文本 | 3 × 3 × 5 = 45 |
| 输出约束 | "请返回 JSON" vs schema 约束 | T=0, effort=medium | 2 × 3 × 5 = 30 |

**小计约 135 次调用**,成本可控(用便宜模型跑)。

### 阶段二:只对"有交互迹象"的组合做交叉

比如阶段一发现"C 类任务在 high effort 下明显更好",再验证"high effort + T=0.7 是否会破坏稳定性"。**只做 2–3 组交叉,不要铺开。**

---

## 4. 指标定义

| 指标 | 怎么测 | 反映哪个轴 |
|---|---|---|
| **准确率** | 对照判定标准 | ②③ |
| **一致性**(关键) | 同一输入跑 N 次,输出的相似度或完全一致率 | ① |
| **多样性** | 不同输入的输出差异度(仅生成任务) | ① |
| **格式合规率** | 能否被 parser 成功解析 | ③ |
| **P50 / P95 延迟** | 端到端计时 | ② |
| **成本** | token 用量 × 单价 | ②③ |

**"一致性"是温度实验的核心指标**——它直接量化"同样的输入会不会给出不同答案"。分类任务的一致性低于 100%,意味着你的评测分数在漂移。

---

## 5. 脚本骨架

```python
import json, time, itertools
from pathlib import Path
from openai import OpenAI

client = OpenAI()  # base_url 可换成任何兼容端点

CONFIGS = [
    # (标签, 温度, 推理强度, 结构化)
    ("T0_mid_free",   0.0, "medium", False),
    ("T0.3_mid_free", 0.3, "medium", False),
    ("T0.7_mid_free", 0.7, "medium", False),
    ("T1.0_mid_free", 1.0, "medium", False),
    ("T0_low_free",   0.0, "low",    False),
    ("T0_high_free",  0.0, "high",   False),
    ("T0_mid_json",   0.0, "medium", True),
]

TASKS = json.loads(Path("tasks.json").read_text())   # [{id, type, input, expected}]
REPEATS = 5

def run_once(cfg, task):
    label, temp, effort, structured = cfg
    kwargs = dict(
        model="你的模型",
        messages=[{"role": "user", "content": task["input"]}],
        temperature=temp,
        # 注意:推理强度参数名各家不同,按你的 provider 调整
        # reasoning_effort=effort,
    )
    if structured:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0
    return {
        "label": label, "task_id": task["id"], "task_type": task["type"],
        "output": resp.choices[0].message.content,
        "latency": latency,
        "tokens": resp.usage.total_tokens,
    }

results = []
for cfg in CONFIGS:
    for task in TASKS:
        for i in range(REPEATS):
            try:
                results.append(run_once(cfg, task))
            except Exception as e:
                results.append({"label": cfg[0], "task_id": task["id"], "error": str(e)})

Path("results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
print(f"完成 {len(results)} 次调用")
```

**然后写一个统计脚本**,对每个 label 计算:准确率、完全一致率、格式合规率、P50/P95 延迟、平均 token。

---

## 6. 结果表模板

跑完后填这张表(数字是示意):

| 配置 | A类准确率 | A类一致性 | C类准确率 | 格式合规率 | P95延迟 | 平均token |
|---|---|---|---|---|---|---|
| T0 / medium | 92% | **100%** | 68% | — | 3.1s | 420 |
| T0.7 / medium | 90% | 76% | **71%** | — | 3.0s | 445 |
| T0 / low | 88% | 100% | 55% | — | 1.4s | 180 |
| T0 / high | 92% | 100% | **84%** | — | 9.6s | 1450 |
| T0 / schema | 93% | 100% | — | **99%** | 3.3s | 430 |

从这张表能直接读出三个结论:分类该用 T=0;推理任务 high effort 值得(68%→84%);结构化约束把格式合规率拉到 99%。

---

## 7. 最终产出:项目参数策略表

这才是这次实验的真正交付物:

| 项目环节 | 温度 | 推理强度 | 输出约束 | 理由 |
|---|---|---|---|---|
| 工单分类 | 0 | low | schema | 必须可复现;分类不需要深推理 |
| 字段抽取 | 0 | low | schema | 同上 |
| 草稿生成 | 0.3–0.7 | medium | 自由文本 + 引用字段 | 要稳定但避免千篇一律 |
| 校验层判定 | 0 | medium | schema | 判定必须确定 |
| 合成训练数据 | 0.9–1.2 | medium | 自由文本 | 需要多样性覆盖长尾 |
| 对抗测试样本 | 1.0 | high | 自由文本 | 目的是生成刁钻输入 |

**这张表会直接进你的网关设计**(第 19–21 周):网关按任务类型自动套用不同采样策略,而不是所有请求都用一套默认参数。

---

## 8. 三个常见陷阱

1. **T=0 不等于绝对确定**。GPU 浮点非确定性、batch 组成差异都会造成极小概率的输出差异。要严格复现,还需要固定随机种子并接受同一批次内的差异
2. **temperature 和 top_p 不要同时调**。两者都作用于同一件事,同时改会无法归因。**固定其中一个,只调另一个**
3. **拿贵的模型做参数探索是浪费**。参数规律在小模型上同样成立,先用便宜模型扫出结论,只在最终验证时用目标模型

---

## 9. 与准则的对应

- 第 2 条(成功标准):每条任务必须预先定义判定标准,否则实验无意义
- 第 7 条(反例进评测集):实验里表现差的配置是"反例",记录进策略表的"不推荐"栏
- 第 9 条(定期做减法):跑完 135 次后,砍掉明显无区分度的配置,别把所有组合都留在报告里
