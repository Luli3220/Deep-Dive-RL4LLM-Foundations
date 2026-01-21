# PPO 

在前文中我们详细介绍了优势函数的计算，可是优势函数只是一个具体的数值，其可以看作为我们具体策略的一个权重。
在正式介绍PPO之前，我们先介绍一下之前的策略优化算法以及其局限性。

## 1.1 策略梯度 (Policy Gradient, PG)

策略梯度的核心思想非常朴素：如果一个动作表现好（优势函数为正），就增加它的出现概率；反之则降低。
其目标函数通常表示为：

$$
J(\theta) = \hat{\mathbb{E}}_t[ \log \pi_{\theta}(a_t|s_t) \hat{A}_t ]
$$

其梯度表示为：

$$
\nabla_{\theta} J(\theta) = \hat{\mathbb{E}}_t\left[\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)\,\hat{A}_t\right]
$$


**PPO虽然简单，但是存在下面几个问题**：

1. **样本利用率极低**：从公式可以看出，它是一个 On-policy 的算法（是基于当前的策略优化的），每条样本只能用一次，用完就丢（后面介绍的重要性采样可以解决这个问题）
2. **训练极其不稳定，方差大**：从下面展开的公式可以看出来，如果智能体偶然尝试了一个概率极低（比如 0.001）的动作，并且这个动作恰好得到了一个还不错的奖励。那么此时梯度放大的倍数将极大。

$$
\nabla_{\theta} J(\theta) = \hat{\mathbb{E}_t} \left[ \underbrace{\frac{1}{\pi_{\theta}(a_t \mid s_t)}} \cdot \underbrace{\nabla_{\theta} \pi_{\theta}(a_t \mid s_t)} \cdot \hat{A}_t \right]
$$


---

## 1.2 信任区域方法 (Trust Region Methods) 

为了解决 PG 的不稳定问题，后续工作提出了以 **TRPO** 为代表的信任区域方法。其核心思路是：**在更新策略时，限制新旧策略之间的差异，确保更新在一定区间内进行。**

其数学表达式为:

$$
\max_{\theta } \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t \right] 
$$

$$
\text{subject to } \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t)]] \leq \delta. 
$$

其中， $\theta_{\text{old}}$ 是更新前的策略参数。在对目标进行线性近似、对约束进行二次近似之后，可以使用共轭梯度算法高效地近似求解该问题（求解很复杂，这里就不详细介绍，后续算法也与此无关，了解即可）。

实际上建议使用惩罚项而非约束，即求解如下的无约束优化问题：

$$
\max_{\theta } \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t)] \right] 
$$

### TRPO 引入的两个核心组件


#### 1. 重要性采样 (Importance Sampling)
* 它是一种统计学技巧，允许我们通过从分布 $q$ （旧策略）中采样的样本，来估计分布 $p$ （当前策略）下的期望。其核心数学形式为([详细推导见](#重要采样))：
  
  $$
  \mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]
  $$

可以观察到，这里将基于 $p$  的概率分布的期望转换为基于 $q$ 的概率分布的期望
等价于 ( $\pi_{\theta}(a_t,s_t)$ ) $->$  ( $\pi_{\theta _{old}}(a_t,s_t)$ ),这样我们就可以用老的样本来更新我们的策略了！

**举个例子：LLM 场景下的直观理解**

在 LLM 的训练中，重要性采样实际上解决了“一边生成一边学习”效率太低的问题。我们可以这样理解这个过程：

1. **采样阶段 (Rollout)**：
   我们先让“旧模型” ( $\pi_{\text{old}}$ ) 去推理生成一段文本（例如回答一个问题）。在这个过程中，我们要记录下旧模型生成每一个 Token 的**概率**。

2. **评估阶段 (Evaluation)**：
   现在我们要优化“新模型” ( $\pi_{\theta}$ )。我们不需要让新模型重新去生成文本，而是让它去看刚才旧模型生成的**那串 Token 序列**，并计算：如果让新模型来写，它在每一个位置写出那个相同 Token 的**概率**是多少。

3. **计算比值 (Ratio)**：
   我们将 **新模型概率 / 旧模型概率** 得到一个比值 $r_t(\theta)$ ：
   * 如果 $r_t(\theta) > 1$ ：说明新模型比旧模型更倾向于说出这个 Token。
   * 如果 $r_t(\theta) < 1$ ：说明新模型在这一点上变得更保守了。

4. **形成损失**：
   我们将这个比值乘以该 Token 对应的**优势函数 $\hat{A}_t$ **（即这个词说得好不好的得分）。



它解决了 **On-policy 效率低**的问题。
* 在原生 PG 中，必须用最新策略 $\pi_\theta$ 采样。

* 引入重要性采样后，我们可以使用旧策略 $\pi_{\theta_{\text{old}}}$ 采集的数据（即 Ratio $r_t(\theta) = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$），这使得数据可以被重复利用（Off-policy 风格，但仍是一个 on-policy 算法）。



#### **2. KL 散度 (KL Divergence)**
* 它衡量两个概率分布之间“距离”的非对称指标。

    $$\text{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$$

在上文的TRPO中，我们减去一个 $\text{KL}[\pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t)]$ 也就是让我们每一次更新和就旧策略偏离不要太远，限制更新幅度不要太大。



虽然TRPO解决了PO算法一些问题，但仍然存在下面几个问题：
1. 二阶导数(**最致命的问题**) : TRPO 为了精确控制 KL 散度，需要计算目标函数的二阶导数,海森矩阵的大小是参数量 $N$ 的平方 ($N \times N$) 计算量巨大。
2. 实现难度极高：相比于只需要几行代码就能实现的 Adam 优化器或 PPO，TRPO 的实现逻辑非常复杂，涉及 Fisher 信息矩阵的近似计算、线搜索（Line Search）等。

---


## 1.3 Clipped Surrogate Objective



## 详细推导

### 重要采样

**推导过程**

**展开期望定义**：将期望写成积分形式：

   $$
   \mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x) \, dx
   $$

**引入参考分布 $q$**：在被积函数中同时乘以并除以 $q(x)$（前提是在 $p(x)f(x) \neq 0$ 的区域内 $q(x) > 0$）：

   $$
   \mathbb{E}_{x \sim p}[f(x)] = \int \frac{q(x)}{q(x)} p(x) f(x) \, dx
   $$

**替换新的概率函数**：将 $q(x)$ 提出来作为新的概率测度（积分基准），剩下的部分看作新的被积函数：

   $$
   \mathbb{E}_{x \sim p}[f(x)] = \int q(x) \left[ \frac{p(x)}{q(x)} f(x) \right] \, dx
   $$

**转换回期望形式**：根据期望定义，我们得到最终形式：
   $$
   \mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]
   $$
