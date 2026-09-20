# Transformer 手感重建 + nanoGPT 代码精读指南

> 配套源码:[nanogpt-model.py](nanogpt-model.py)(共 330 行,本文所有 L 行号都指它)
> 方法:先读直觉,再读代码,最后合上书自己写伪代码。三遍法,缺一不可。

---

## 0. 全局形状:GPT.forward 全景(L170–L192)

先建立整张地图,所有细节都挂在这根主轴上:

```
idx [B, T]                      整数 token 序列,B=batch, T=序列长度
  ↓ wte 查表                     token embedding
  ↓ wpe 查表                     position embedding([T, C],广播加到每个样本上)
x [B, T, C]                     C=n_embd,每个位置的"语义坐标"
  ↓ × N 个 Block                 N=n_layer,如 12
x [B, T, C]                     形状从头到尾不变!(残差结构的意义)
  ↓ ln_f                         最后的 LayerNorm
  ↓ lm_head                      Linear(C → vocab_size)
logits [B, T, vocab]            每个位置各给一个"下一个 token"的分布
  ↓ cross_entropy(logits, targets 右移一位)
loss 标量
```

三个立刻值得注意的工程细节:

1. **targets 就是 idx 右移一位**——LM 训练不需要人工标注,语料自己就是标签。这是"自监督"的全部含义
2. **L138 weight tying**:输入 embedding 表和输出 lm_head 共享同一个权重。直觉:一个 token "长什么样"和"该被预测成什么样"是同一套语义,共享省一倍参数还更稳
3. **L186–L188 推理优化**:推理时只取最后一个位置过 lm_head(`x[:, [-1], :]`),因为只需要下一个 token 的分布,前面 T-1 个位置的输出白算。省的是 vocab 维度的矩阵乘,很可观

---

## 1. 为什么 MLP 必须被换掉

回看上一课的 MLP:拼接固定窗口 → 过权重。三个死穴:

1. **窗口固定**:只能看 3 个就看 3 个,第 4 个字符之前的信息永远丢了
2. **权重与位置绑定**:W1 的列对应"位置 1 的字符 × 位置 2 的字符"这种固定搭配,换个词序就要重新学
3. **内容无关的路由**(最致命):拼接后,前面字符对预测的贡献权重是**写死在 W1 里的**。但直觉上,该看哪个词应该由**内容**决定——"The animal didn't cross the street because it was too tired"里,"it"该主要看 "animal",这个判断只能看了内容才能做

Attention 的回答:**让每个位置根据自己的内容,动态决定去看哪些位置、各看多少。**

---

## 2. Self-Attention 机制(形状视角,L52–L75)

### 三个角色

每个位置的向量 x 被同一次投影变成三份(L56,一次 Linear 出 3C 再切开):

```
Q (query):  我在找什么    —— 主动方,发出"请求"
K (key):     我是什么      —— 被动方,挂出"门牌/索引"
V (value):   我携带什么    —— 被动方,真正被取走的"内容"
```

### 形状流(单样本直觉 → batch 实况)

```
x        [B, T, C]
  ↓ c_attn (Linear C→3C), split 成 q,k,v
q,k,v    [B, T, C]
  ↓ view + transpose:拆成多头,nh 个头,每个头 hs = C/nh 维
q,k,v    [B, nh, T, hs]
  ↓ q @ k.transpose(-2,-1) / sqrt(hs)
att      [B, nh, T, T]   ← 这就是那张"谁看谁"的图!
  ↓ causal mask:上三角填 -inf
  ↓ softmax(dim=-1):每行变成一个和为 1 的分布
att      [B, nh, T, T]   att[b][h][i][j] = 位置 i 分给位置 j 的注意力比例
  ↓ att @ v
y        [B, nh, T, hs]  位置 i 的新表示 = 所有位置 v 的加权和
  ↓ transpose + view:多头拼回去
y        [B, T, C]
  ↓ c_proj (Linear C→C)
输出      [B, T, C]
```

### 用你"维度即图像"的习惯来渲染

att 那张 [T, T] 的图就是核心产物,值得盯着看:

- **第 i 行**:位置 i 的注意力分配——这一行 softmax 后和为 1
- **第 i 行第 j 列大**:预测位置 i 的下一个词时,重点参考位置 j 的内容
- **causal mask 后**:上三角全 0,图是**下三角**——位置 i 只能看 ≤ i 的位置,不许偷看未来(否则训练时答案就泄漏了,这就是 "causal" 的含义)

### 三个容易被略过的"为什么"

1. **为什么除以 √hs(L67)**:q·k 是 hs 个数的和,方差随 hs 线性增长。不除的话,维度一大,打分分布的尺度就膨胀,softmax 被顶到饱和区(又是饱和!和 dead tanh 同源),梯度消失。除以 √hs 让打分方差回到 1 附近
2. **为什么 mask 填 -inf 而不是 0(L68)**:softmax 是 exp 之后归一化,exp(-inf)=0 才是真正"零权重";填 0 的话 exp(0)=1,照样分到注意力
3. **为什么要多头(L57–L59)**:一个头只有一套 QK 投影,只能学一种"匹配模式"。多头相当于并行的多组探测器——一个头盯语法依存,一个头盯指代,一个头盯位置邻近。拆维度(每头 hs=C/nh)而不是加参数,总计算量不变

---

## 3. Block:残差 + pre-norm + FFN(L94–L107)

```python
x = x + self.attn(self.ln_1(x))   # L104: 先归一化,再注意力,加回残差
x = x + self.mlp(self.ln_2(x))    # L105: 同上,换成 FFN
```

### 为什么是 `x + f(x)` 而不是 `f(x)`

残差连接让梯度有一条"高速公路"直通浅层——没有它,12 层堆起来梯度早就消失殆尽了(回忆 tanh 饱和的教训)。同时它让每层只需要学"增量修改",而不是从头重写表示。CV 里 ResNet 的思想,原封不动搬过来的。

### pre-norm 的顺序讲究

原始 Transformer 是 post-norm(`ln(x + f(x))`),现代 GPT 全是 pre-norm(`x + f(ln(x))`)。原因:pre-norm 下残差主干从输入到输出全程不被 LayerNorm 打扰,梯度路径干净,深网络不需要warmup 也能稳定训练。

### MLP(L78–L91)——你的老朋友

```
C → 4C (c_fc) → GELU → C (c_proj)
```

这就是你上一轮刚吃透的结构,连"升维加工、降维输出"的形状都一模一样,只是升维倍数固定在 4x。一个 Block 的分工:**attention 负责位置之间的信息搬运,MLP 负责每个位置内部的信息加工**。前者是"通信",后者是"计算"。

### 关于归一化,呼应你的错题

Transformer 用 LayerNorm 而不是 BatchNorm,恰恰是冲着 BatchNorm 那两个毛病去的:LayerNorm 在**单个样本的特征维度**上归一化(对 C 维求均值方差),不碰 batch 维度——训练和推理行为完全一致,样本之间零耦合。

---

## 4. 伪代码练习(核心作业)

**规则:合上书,先自己写,写完再对照 L52–L75。** 手写一遍的记忆强度是读三遍的三倍。

### 题目

用伪代码写出 causal self-attention 的 forward,输入 x [B, T, C],要求:
体现 QKV 投影、多头拆分、打分缩放、因果 mask、softmax、加权求和、多头合并、输出投影。

### 参考答案(写完再看)

```
function causal_self_attention(x):          # x: [B, T, C]
    qkv = Linear_C_to_3C(x)                 # [B, T, 3C]
    q, k, v = split(qkv, 3 份)              # 各 [B, T, C]

    for each head h in 1..nh:               # 实现上用 view+transpose 并行,不写循环
        q_h, k_h, v_h = 取第 h 段 [hs 维]    # [B, T, hs]

        score = q_h @ k_h^T / sqrt(hs)      # [B, T, T] 匹配度矩阵
        score[上三角] = -inf                 # 因果 mask:不许看未来
        att = softmax(score, dim=-1)        # 每行和为 1
        y_h = att @ v_h                     # [B, T, hs] 加权求和

    y = concat(y_1..y_nh)                   # [B, T, C] 多头拼回
    return Linear_C_to_C(y)                 # 输出投影
```

对照时重点检查:缩放有没有写、mask 的位置(必须在 softmax 之前)、矩阵乘的顺序(att @ v 而不是 v @ att)。

---

## 5. 代码精读任务清单(带行号,逐条回答)

读 [nanogpt-model.py](nanogpt-model.py),每题在代码里找到答案,并写下自己的一句话解释:

1. **L34**:为什么 QKV 用一个 `Linear(C, 3C)` 而不是三个独立 Linear?(提示:一次矩阵乘 vs 三次,GPU 喜欢哪个?)
2. **L57–L59**:`view(B, T, nh, hs).transpose(1, 2)` 这两步分别在做什么?为什么 transpose 之后矩阵乘就是"每个头独立计算"?
3. **L48**:不用 flash attention 时,`register_buffer` 注册的下三角 mask 是什么?为什么不注册成 parameter?
4. **L72**:多头合并为什么 transpose 回来还要 `.contiguous()` 才能 view?
5. **L104–L105**:把这两行改成 `x = self.attn(x)`(去掉残差)会发生什么?改成 post-norm 呢?
6. **L142–L143**:为什么残差路径上的投影层(c_proj)初始化要额外除以 √(2·n_layer)?(提示:12 层残差累加,方差怎么长?)
7. **L138**:weight tying 省了多少参数?用 vocab=50304、C=768 算个数。
8. **L186–L188**:推理时 `x[:, [-1], :]` 省了多少计算?为什么训练时不能这么省?
9. **L64–L65**:flash attention 快在哪里?省的是时间还是显存?(这题允许查资料,答案和 KV cache 无关,和 [T,T] 矩阵是否显式落显存有关)

---

## 6. 自测(不看代码能答才算过)

1. att 矩阵 [T, T] 的第 i 行是什么含义?为什么每行和为 1?
2. 如果没有 causal mask,训练时会发生什么泄漏?
3. Q 和 K 如果不分成两个矩阵,共用一个,模型会失去什么能力?
4. 一个 Block 里,attention 和 MLP 各负责什么?为什么 MLP 要升维到 4C 再降回来?
5. pre-norm 相比 post-norm,梯度路径上少了什么障碍?
6. 为什么推理可以只算最后一个位置的 logits,训练不行?
7. 把 block_size 从 256 加到 1024,att 矩阵的显存占用涨几倍?(提示:T²)
