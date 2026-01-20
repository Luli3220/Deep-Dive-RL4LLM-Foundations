# RL foundation

在学习 RL4LLM 之前，需要简单了解一些 RL 基础的内容，比如环境、智能体、奖励函数等。其本质是一个 **智能体（Agent）** 通过与 **环境（Environment）** 不断交互，以最大化 **累积奖励（Cumulative Reward）** 的过程。

## 目录

- [关键术语](#key-terms)
  - [1.1 智能体（Agent）](#agent)
  - [1.2 环境（Environment）](#environment)
  - [1.3 状态（State）](#state)
  - [1.4 动作（Action）](#action)
  - [1.5 奖励（Reward）](#reward)
  - [1.6 策略（Policy）](#policy)
- [RL优化目标](#rl-objectives)
  - [2.1 回报 $G_t$ (Return)](#return)
  - [2.2 状态值函数(State-Value Function)](#state-value)
  - [2.3. 动作价值函数 (Action-Value Function)](#action-value)
  - [2.4. 优势函数 Advantage Function](#advantage)
- [贝尔曼方程详细推导](#bellman-derivations)
  - [$V_{\pi}(s_t)$](#V-derivation)
  - [$Q_{\pi}(s_t, a_t)$](#q-derivation)
- [参考资料](#references)

<a id="key-terms"></a>
## 关键术语

<a id="agent"></a>
### 1.1 智能体（Agent）

**智能体（Agent）** 是在环境中执行动作、并通过优化算法不断调整自身行为，以最大化 **累积奖励** 的主体。

在经典 RL 中，智能体可能是一个机器人或控制器；
**在 RL4LLM 中，智能体通常是一个（LLM）**。

* **动作**：在给定上下文下生成下一个 Token
* **目标**：最大化生成序列的期望奖励

从形式上看，LLM 定义了一个参数化策略：
$$
\pi_\theta(a_t \mid s_t)
$$

表示在状态 $ (s_t) $ （当前文本上下文）下，生成 Token $(a_t) $的概率。 

---

<a id="environment"></a>
### 1.2 环境（Environment）

**环境（Environment）** 是智能体所处的外部系统，它接收动作、更新状态，并返回奖励。

在 RL4LLM 中，环境可以抽象为一个**文本生成过程**：

* 智能体生成一个 Token
* 环境将该 Token 拼接到当前序列中
* 状态（上下文）随之更新
* 在序列结束后（或中间），环境给出奖励

整个交互过程可以写为：
$$
s_{t+1} = \text{Concat}(s_t, a_t)
$$

---

<a id="state"></a>
### 1.3 状态（State）

**状态（State）** 是环境在某一时刻的描述，是智能体做出决策的依据。

在 RL4LLM 中，状态通常定义为：

**Prompt + 已生成的 Token 序列**
即： 
$$
s_t = (x, a_1, a_2, \dots, a_{t-1})
$$

其中 (x) 是用户输入的 Prompt。

---

<a id="action"></a>
### 1.4 动作（Action）

**动作（Action）** 是智能体在特定状态下可执行的行为。

在 RL4LLM 中：

* 动作空间是 **词表 $( \mathcal{V} ) $**
* 每一步的动作是选择一个 Token：
  $$
  a_t \in \mathcal{V}
  $$


---

<a id="reward"></a>
### 1.5 奖励（Reward）

**奖励（Reward）** 是环境对智能体行为的反馈，用于衡量动作或序列的好坏。

在 RL4LLM 中，常见奖励来源包括：

* **奖励模型（Reward Model, RM）**
* **基于规则的奖励（RLVR）**

例如，在数学推理任务中：
$$
r =
\begin{cases}
+1, & \text{答案正确} \\
0,  & \text{答案错误}
\end{cases}
$$

---

<a id="policy"></a>
### 1.6 策略（Policy）

**策略（Policy）** 定义了智能体在不同状态下如何选择动作。

在 RL4LLM 中，策略就是 **LLM 的参数化概率分布**：
$$
\pi_\theta(a_t \mid s_t) = P_\theta(\text{next token} = a_t \mid \text{context } s_t)
$$

训练的核心目标是找到最优参数：
$$
\theta^* = \arg\max_\theta \; \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
$$
其中：

* $(\tau)$ 是生成的一整个 Token 序列
* $(R(\tau))$ 是该序列的总奖励

---


<a id="rl-objectives"></a>
## RL优化目标 

<a id="return"></a>
### 2.1 回报 $G_t$ (Return)

回报 $G_t$：智能体在某一次具体尝试中，从t时刻开始到结束，最终获得的总“分数”，即所有奖励的加权折扣和。 公式如下：

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}
$$

- γ是折扣因子。0<γ<1。它体现了智能体对未来的“远见”程度。γ 越接近0，智能体越“短视”，只关心眼前的奖励；γ越接近1，智能体越“有远见”，未来的奖励和眼前的奖励几乎同等重要。
- $R(s_t,a_t)$ : 单步奖励，在当前状态 $s_t$ 下，选择某个动作 $a_t$ 获得的即时奖励

<a id="state-value"></a>
### 2.2 状态值函数(State-Value Function)


状态值函数 $V_{\pi}(s_t)$ 是在状态 $s_t$ 下，期望的回报。它表示在当前状态下，如果按照策略行动，未来能获得多少奖励。数学上表示为：

$$
V_{\pi}(s_t) = \mathbb{E}_{\pi}[G_t \mid S_t = s_t]
$$

展开期望计算([详细推导](#V-derivation))：

$$V_{\pi}(s_t) = \mathbb{E}_{\pi} [R_t + \gamma G_{t+1} \mid S_t = s_t] = \mathbb{E}_{\pi} [R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t]$$

这个递归形式被称为 **贝尔曼方程 (Bellman equation)**，用于计算状态值函数。


<a id="action-value"></a>
### 2.3. 动作价值函数 (Action-Value Function)

动作价值函数 $Q(s_t, a_t)$ 衡量在状态 $s_t$ 下采取特定动作 $a_t$ 后，按照策略 $\pi$ 继续执行时的期望回报：

$$Q_{\pi}(s_t, a_t) = \mathbb{E}_{\pi}[G_t \mid S_t = s_t, A_t = a_t]$$

这表示在状态 $s_t$ 采取动作 $a_t$ 后，未来的累积回报的期望。

同样展开期望([详细推导](#q-derivation))：

$$Q_{\pi}(s_t, a_t) = \mathbb{E}_{\pi} [R_t + \gamma G_{t+1} \mid S_t = s_t, A_t = a_t] = \mathbb{E}_{\pi} [R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t, A_t = a_t]$$

这表示在状态 $s_t$ 采取动作 $a_t$ 后，未来的累积回报的期望。


<a id="advantage"></a>
### 2.4. 优势函数 Advantage Function

动作价值函数与状态值函数的区别在于：$Q(s_t, a_t)$是当前状态 $s_t$ 做出行动 $a_t$后的期望奖励。
由此我们依此得到我们的优势函数：$A(s_t, a_t)$ 衡量某个动作 $a_t$ 在状态 $s_t$ 处比平均值（即状态值）好多少：

$$A_{\pi}(s_t, a_t) = Q_{\pi}(s_t, a_t) - V_{\pi}(s_t)$$

这表示：

* 若 $A_{\pi}(s_t, a_t) > 0$，说明这个动作 $a$ 比策略的平均表现更好，应该更倾向于选择它；
* 若 $A_{\pi}(s_t, a_t) < 0$，说明这个动作 $a$ 比策略的平均表现更差，应该减少选择它的概率。

由状态值与动作值的贝尔曼方程：

$$V_{\pi}(s_t) = \mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t]$$

$$Q_{\pi}(s_t, a_t) = \mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t, A_t = a_t]$$

两式相减即可得到优势函数的等价形式：

$$A_{\pi}(s_t, a_t) = \mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t, A_t = a_t] - \mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t]$$

由于 $V_{\pi}(s) = \mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s]$，上式也可写为：

$$A_{\pi}(s_t, a_t) = \mathbb{E}_{\pi}[R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t, A_t = a_t] - V_{\pi}(s_t)$$

这样我们就可以得到一个只依赖于状态价值函数的一个优化目标了。

以上就是我们RL中最核心的优势函数的推导了，这也是后续 PPO 等 Actor-Critic 方法里 Critic 拟合 $V_{\pi}(s)$、Actor 用 $A_{\pi}(s, a)$ 更新策略的基础。



<a id="bellman-derivations"></a>
## 贝尔曼方程详细推导

<a id="V-derivation"></a>
### $V_{\pi}(s_t)$

$$
V_{\pi}(s_t) = R(s_t) + \gamma \mathbb{E}_{\pi}[V_{t+1} \mid S_t = s_t]
$$

### 推导如下:

#### 首先将$V(s_t)$展开
$$
\begin{aligned}
V(s_t) &= \mathbb{E}[G_t \mid S_t = s_t] \quad \text{（状态价值函数的定义）} \\
&= \mathbb{E}[R_t + \gamma G_{t+1} \mid S_t = s_t] \quad \text{（回报的定义）} \\
&= \mathbb{E}[R_t \mid S_t = s_t] + \gamma \mathbb{E}[G_{t+1} \mid S_t = s_t] \quad \text{（期望的可加性）} \\
&= R(s_t) + \gamma \mathbb{E}[G_{t+1} \mid S_t = s_t] \quad \text{（回报的定义）}
\end{aligned}
$$

#### 建立状态价值之间的联系

* 首先从定义上可以显然知道两者的关系，即 $V(s_{t+1}) = \mathbb{E} [G_{t+1} \mid S_{t+1} = s_{t+1}]$。这里我们发现其实已经很接近上面的 $\mathbb{E} [G_{t+1} \mid S_t = s_t]$ 了，两者求期望的目标都是 $G_{t+1}$，只不过两者求期望的条件不一样，一个是 $s_t$，一个是 $s_{t+1}$。这个时候我们注意，$s_{t+1}$ 就是 $s_t$ 的下一个状态，这时候我们就可以用**全期望公式**建立起它们的联系了。

* **全期望公式：**
  $$\mathbb{E}(X) = \sum_k^m P(Y_k) \mathbb{E}(X \mid Y_k) = \mathbb{E}(\mathbb{E}(X \mid Y))$$

那么由全期望公式我们可以得到：

$$\mathbb{E}[G_{t+1} \mid S_t = s_t] = \sum_{s_{t+1}} \mathbb{E}[G_{t+1} \mid S_t = s_t, S_{t+1} = s_{t+1}] p(s_{t+1} \mid s_t)$$

再由于马尔可夫性，$G_{t+1}$ 其实只和 $s_{t+1}$ 有关，和 $s_t$ 无关，可以理解为 **（未来的状态只取决于现在，而与过去无关）** 因此：

$$\mathbb{E}[G_{t+1} \mid S_t = s_t, S_{t+1} = s_{t+1}] = \mathbb{E}[G_{t+1} \mid S_{t+1} = s_{t+1}] = V(s_{t+1})$$

将其代回上一个式子得，

$$\mathbb{E}[G_{t+1} \mid S_t = s_t] = \sum_{s_{t+1}} V(s_{t+1}) p(s_{t+1} \mid s_t) = \mathbb{E}[V_{t+1} \mid S_t = s_t]$$

综上

$$
\begin{aligned}
V(s_t) &= R(s_t) + \gamma \mathbb{E}[G_{t+1} \mid S_t = s_t] \\
&= R(s_t) + \gamma \sum_{s_{t+1}} V(s_{t+1}) p(s_{t+1} \mid s_t) \\
&= R(s_t) + \gamma \mathbb{E}[V_{t+1} \mid S_t = s_t]
\end{aligned}
$$

<a id="q-derivation"></a>
### $Q_{\pi}(s_t, a_t)$

$$
Q_{\pi}(s_t, a_t) = R(s_t, a_t) + \gamma \mathbb{E}_{\pi}[V_{\pi}(S_{t+1}) \mid S_t = s_t, A_t = a_t]
$$

### 推导如下:

#### 首先将$Q(s_t, a_t)$展开
$$
\begin{aligned}
Q(s_t, a_t) &= \mathbb{E}[G_t \mid S_t = s_t, A_t = a_t] \quad \text{（动作价值函数的定义）} \\
&= \mathbb{E}[R_t + \gamma G_{t+1} \mid S_t = s_t, A_t = a_t] \quad \text{（回报的定义）} \\
&= \mathbb{E}[R_t \mid S_t = s_t, A_t = a_t] + \gamma \mathbb{E}[G_{t+1} \mid S_t = s_t, A_t = a_t] \quad \text{（期望的可加性）} \\
&= R(s_t, a_t) + \gamma \mathbb{E}[G_{t+1} \mid S_t = s_t, A_t = a_t] \quad \text{（回报的定义）}
\end{aligned}
$$

#### 建立动作价值之间的联系

* 同理把 $\mathbb{E}[G_{t+1} \mid S_t = s_t, A_t = a_t]$ 用全期望公式按下一状态 $S_{t+1}$ 展开：
  $$\mathbb{E}[G_{t+1} \mid S_t = s_t, A_t = a_t] = \sum_{s_{t+1}} \mathbb{E}[G_{t+1} \mid S_t = s_t, A_t = a_t, S_{t+1} = s_{t+1}] p(s_{t+1} \mid s_t, a_t)$$

* 由马尔可夫性，给定 $S_{t+1}$ 后，未来回报 $G_{t+1}$ 与过去 $(S_t, A_t)$ 无关，因此：
  $$\mathbb{E}[G_{t+1} \mid S_t = s_t, A_t = a_t, S_{t+1} = s_{t+1}] = \mathbb{E}[G_{t+1} \mid S_{t+1} = s_{t+1}] = V(s_{t+1})$$

将其代回上一个式子得，

$$\mathbb{E}[G_{t+1} \mid S_t = s_t, A_t = a_t] = \sum_{s_{t+1}} V(s_{t+1}) p(s_{t+1} \mid s_t, a_t) = \mathbb{E}[V_{t+1} \mid S_t = s_t, A_t = a_t]$$

因此可以得到一个常用的形式：

$$
\begin{aligned}
Q(s_t, a_t) &= R(s_t, a_t) + \gamma \sum_{s_{t+1}} V(s_{t+1}) p(s_{t+1} \mid s_t, a_t) \\
&= R(s_t, a_t) + \gamma \mathbb{E}[V_{t+1} \mid S_t = s_t, A_t = a_t]
\end{aligned}
$$

代入即可得到动作价值的贝尔曼期望方程：

$$
Q_{\pi}(s_t, a_t) = \mathbb{E}_{\pi} [R_t + \gamma V_{\pi}(S_{t+1}) \mid S_t = s_t, A_t = a_t]
$$ 

<a id="references"></a>
## 参考资料

- https://misaka0502.github.io/2025/03/14/return-value-q-advantage-in-rl/
- https://blog.csdn.net/qq_41936559/article/details/142644560




