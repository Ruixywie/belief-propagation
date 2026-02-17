# Belief Propagation: Principles, Algorithms, and Applications
信念传播：原理、算法和应用

* * *

## Table of Contents
目录

1.  [Introduction
    介绍](#1-introduction)
2.  [Probabilistic Graphical Models
    概率图模型](#2-probabilistic-graphical-models)
3.  [Factor Graphs
    因子图](#3-factor-graphs)
4.  [Message Passing
    消息传递](#4-message-passing)
5.  [The Sum-Product Algorithm
    和积算法](#5-the-sum-product-algorithm)
6.  [The Max-Product Algorithm
    最大乘积算法](#6-the-max-product-algorithm)
7.  [Exact Inference on Trees
    树上的精确推理](#7-exact-inference-on-trees)
8.  [Loopy Belief Propagation
    循环信念传播](#8-loopy-belief-propagation)
9.  [Numerical Example
    数值示例](#9-numerical-example)
10.  [Applications
    应用程序](#10-applications)
11.  [Conclusion
    结论](#11-conclusion)
12.  [References
    参考](#12-references)

* * *

## 1\. Introduction
1\. 引言

Probabilistic inference is a cornerstone of modern machine learning, statistics, and artificial intelligence. Given a joint probability distribution over a set of random variables, we often wish to answer queries such as:
概率推理是现代机器学习、统计学和人工智能的基石。给定一组随机变量的联合概率分布，我们常常希望回答如下问题：

*   **Marginal inference**: What is the probability of a single variable $x_i$ taking a particular value, after summing out all other variables?
    **边际推断** ：单个变量 x 的概率是多少？ 𝑖 x i ​ 在将所有其他变量相加之后，取某个特定值？
*   **MAP inference**: What is the most probable joint assignment to all variables?
    **最大后验概率 (MAP) 推断** ：所有变量最可能的联合赋值是什么？
*   **Conditional inference**: What is the posterior distribution of some variables given observed evidence?
    **条件推断** ：给定观测证据，某些变量的后验分布是什么？

For models with many variables, exact computation of these quantities involves summing or maximizing over an exponentially large state space, making brute-force enumeration intractable. **Belief Propagation (BP)** provides an elegant and efficient framework for performing these computations by exploiting the structure of the underlying probabilistic graphical model.
对于具有众多变量的模型，精确计算这些量需要在指数级庞大的状态空间上进行求和或最大化，这使得穷举法难以实现。 **置信传播（BP）** 通过利用底层概率图模型的结构，提供了一个优雅而高效的框架来执行这些计算。

Originally introduced by Judea Pearl in 1982 for tree-structured models, Belief Propagation is a **message-passing algorithm** that computes exact marginals on trees and provides powerful approximations on general graphs. It has become one of the most widely used algorithms in probabilistic reasoning, with applications spanning error-correcting codes, computer vision, natural language processing, and computational biology.
信念传播算法最初由 Judea Pearl 于 1982 年提出，用于树状结构模型。它是一种**消息传递算法** ，能够计算树上的精确边缘分布，并为一般图提供强大的近似解。如今，它已成为概率推理领域应用最广泛的算法之一，其应用涵盖纠错码、计算机视觉、自然语言处理和计算生物学等领域。

This report provides a comprehensive introduction to Belief Propagation, starting from the foundations of probabilistic graphical models and building up to the Sum-Product and Max-Product algorithms, with both theoretical exposition and visual illustrations.
本报告全面介绍了信念传播，从概率图模型的基础知识入手，逐步深入到求和积算法和最大积算法，并辅以理论阐述和视觉示例。

* * *

## 2\. Probabilistic Graphical Models
2\. 概率图模型

A **Probabilistic Graphical Model (PGM)** represents a joint probability distribution using a graph, where nodes correspond to random variables and edges encode conditional dependencies or interactions. PGMs come in three main flavors:
**概率图模型 (PGM)** 使用图来表示联合概率分布，其中节点对应于随机变量，边编码条件依赖关系或交互作用。PGM 主要有三种类型：

![Probabilistic Graphical Models Overview](media/images/bp_scenes/PGMOverview_ManimCE_v0.19.2.png)

### 2.1 Bayesian Networks (Directed Models)
2.1 贝叶斯网络（有向模型）

A **Bayesian Network** (BN) is a directed acyclic graph (DAG) where each node $x_i$ is associated with a conditional probability distribution given its parents:
**贝叶斯网络** （BN）是一个有向无环图（DAG），其中每个节点𝑥 𝑖 x i ​ 与给定其父母的条件概率分布相关：

$p(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \mid \text{pa}(x_i))$

The directed edges encode causal or generative relationships. For example, in a medical diagnosis model, a disease node might point to symptom nodes, representing that the disease *causes* the symptoms.
有向边编码因果关系或生成关系。例如，在医学诊断模型中，疾病节点可能指向症状节点，表示该疾病*导致了*这些症状。

**Key properties:
主要特性：**

*   Encodes conditional independencies via the *d-separation* criterion
    通过 *d 分离*准则对条件独立性进行编码
*   Naturally represents causal/generative processes
    自然地代表因果/生成过程
*   Parameters are conditional probability tables (CPTs)
    参数是条件概率表（CPT）。

### 2.2 Markov Random Fields (Undirected Models)
2.2 马尔可夫随机场（无向模型）

A **Markov Random Field** (MRF), also called an undirected graphical model, uses an undirected graph where the joint distribution factorizes over cliques:
**马尔可夫随机场** （MRF），也称为无向图模型，使用无向图，其中联合分布在团上分解：

$p(x_1, x_2, \ldots, x_n) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \psi_c(\mathbf{x}_c)$

where $\psi_c$ are non-negative **potential functions** defined over cliques $c$, and $Z = \sum_{\mathbf{x}} \prod_c \psi_c(\mathbf{x}_c)$ is the **partition function** ensuring normalization.
其中 𝜓 𝑐 ψ c ​ 是定义在团 $c$ 上的非负**势函数** ，而 $Z = \sum_{\mathbf{x}} \prod_c \psi_c(\mathbf{x}_c)$ 是确保归一化的**配分函数** 。

**Key properties:
主要特性：**

*   Encodes symmetric relationships (no notion of directionality)
    编码对称关系（无方向性概念）
*   Conditional independencies follow from graph separation
    条件独立性源于图分离
*   Widely used in image processing and spatial statistics (e.g., Ising model)
    广泛应用于图像处理和空间统计（例如，伊辛模型）

### 2.3 Factor Graphs (Bipartite Representation)
2.3 因子图（二分图表示）

A **Factor Graph** is a bipartite graph with two types of nodes — **variable nodes** and **factor nodes** — connected by edges. It provides a unified and more fine-grained representation that can encode both directed and undirected models. Factor graphs are the natural setting for Belief Propagation, and we discuss them in detail in the next section.
**因子图**是一种二分图，它由两种类型的节点—— **变量节点**和**因子节点** ——通过边连接而成。它提供了一种统一且更细粒度的表示方法，可以编码有向模型和无向模型。因子图是置信传播的天然框架，我们将在下一节详细讨论。

* * *

## 3\. Factor Graphs
3\. 因子图

A factor graph makes the factorization of a probability distribution explicit. Given a joint distribution that factorizes as:
因子图可以明确地表示概率分布的因子分解。给定一个可以分解为以下形式的联合分布：

$p(x_1, x_2, \ldots, x_n) = \prod_{a} f_a(\mathbf{x}_a)$

where each $f_a$ is a **factor** (a non-negative function) that depends on a subset $\mathbf{x}_a$ of variables, the factor graph is constructed as follows:
其中每个 𝑓 𝑎 f a ​ 是一个**因子** （一个非负函数），它依赖于子集𝑥 𝑎 x a ​ 对于所有变量，因子图的构建方式如下：

*   **Variable nodes** (circles): One for each random variable $x_i$
    **变量节点** （圆圈）：每个随机变量 $x_i$ 对应一个节点。
*   **Factor nodes** (squares): One for each factor $f_a$
    **因子节点** （方格）：每个因子对应一个 $f_a$
*   **Edges**: An edge connects variable node $x_i$ to factor node $f_a$ if and only if $x_i \in \mathbf{x}_a$
    **边** ：边连接变量节点𝑥 𝑖 x i ​ 对节点 𝑓 进行因子分析 𝑎 f a ​ 当且仅当 $x_i \in \mathbf{x}_a$

![Factor Graph Structure](media/images/bp_scenes/FactorGraphIntro_ManimCE_v0.19.2.png)

In the figure above, the joint distribution $p(x_1, x_2, x_3, x_4) = f_a(x_1, x_2) \cdot f_b(x_2, x_3) \cdot f_c(x_3, x_4)$ is represented by a chain-like factor graph with variable nodes (blue circles) and factor nodes (red squares).
在上图中，联合分布 $p(x_1, x_2, x_3, x_4) = f_a(x_1, x_2) \cdot f_b(x_2, x_3) \cdot f_c(x_3, x_4)$ 由链状因子图表示，其中变量节点（蓝色圆圈）和因子节点（红色方块）。

### 3.1 Why Factor Graphs?
3.1 为什么需要因式分解图？

Factor graphs offer several advantages:
因子图具有以下几个优点：

1.  **Explicit factorization**: Unlike Bayesian networks or MRFs, the factor graph shows exactly which factors connect which variables, even when multiple factors share the same variable set.
    **显式因子分解** ：与贝叶斯网络或 MRF 不同，因子图准确地显示了哪些因子连接哪些变量，即使多个因子共享同一变量集。
2.  **Unified framework**: Both directed and undirected models can be converted to factor graphs. A Bayesian network's CPTs become factors; an MRF's clique potentials become factors.
    **统一框架** ：有向模型和无向模型均可转换为因子图。贝叶斯网络的 CPT 成为因子；马尔可夫随机场的团势成为因子。
3.  **Natural setting for message passing**: The bipartite structure of factor graphs directly supports the definition of variable-to-factor and factor-to-variable messages.
    **消息传递的自然设置** ：因子图的二分结构直接支持定义变量到因子和因子到变量的消息。

### 3.2 Notation
3.2 符号

Throughout this report, we use the following notation:
本报告中，我们使用以下符号：

| Symbol象征 | Meaning意义 |
| --- | --- |
| xix\_ixi​ | Random variable (variable node)随机变量（变量节点） |
| faf\_afa​ | Factor (factor node)因子（因子节点） |
| N(x)N(x)N(x) | Set of factor nodes neighboring variable xxx变量 xxx 相邻的因子节点集合 |
| N(f)N(f)N(f) | Set of variable nodes neighboring factor fff相邻因子为 fff 的可变节点集合 |
| xa\\mathbf{x}\_axa​ | Set of variables connected to factor faf\_afa​与因子 faf\_afa​ 相关的变量集 |
| μx→f(x)\\mu\_{x \\to f}(x)μx→f​(x) | Message from variable xxx to factor fff来自变量 xxx 的消息给因子 fff |
| μf→x(x)\\mu\_{f \\to x}(x)μf→x​(x) | Message from factor fff to variable xxx来自因子 fff 到变量 xxx 的消息 |
| b(xi)b(x\_i)b(xi​) | Belief (approximate marginal) at variable xix\_ixi​变量 xix\_ixi​ 处的信念（近似边际） |

* * *

## 4\. Message Passing
4\. 消息传递

The core idea of Belief Propagation is **message passing**: nodes in the factor graph exchange local information (messages) with their neighbors, and through iterative exchange, global information about the joint distribution propagates through the network.
信念传播的核心思想是**消息传递** ：因子图中的节点与其邻居交换局部信息（消息），并通过迭代交换，关于联合分布的全局信息在网络中传播。

[https://github.com/user-attachments/assets/placeholder-message-passing](https://github.com/user-attachments/assets/placeholder-message-passing)

> *Animation: Message Passing Mechanism — see `media/videos/bp_scenes/720p30/MessagePassing.mp4`
> 动画：消息传递机制 — 参见 `media/videos/bp_scenes/720p30/MessagePassing.mp4`*

### 4.1 Variable-to-Factor Messages
4.1 变量到因子的信息

A variable node $x$ sends a message to a neighboring factor node $f$ by collecting all incoming messages from its *other* neighboring factors and multiplying them together:
变量节点 $x$ 通过收集来自其*其他*相邻因子的所有传入消息并将它们相乘，向相邻因子节点 $f$ 发送消息：

$\mu_{x \to f}(x) = \prod_{g \in N(x) \setminus f} \mu_{g \to x}(x)$

**Intuition**: The variable $x$ tells factor $f$ everything it has learned from all sources *except* $f$ itself. This prevents information from being "echoed" back to its source.
**直觉** ：变量 $x$ 将它从*除自身以外的*所有来源学到的所有信息告诉因子 $f$ 。这可以防止信息“回传”到其来源。

**Special case — Leaf variable**: If $x$ is a leaf node (connected to only one factor), then $N(x) \setminus f = \emptyset$, and the message is simply:
**特殊情况——叶变量** ：如果 $x$ 是一个叶节点（仅连接到一个因子），则 $N(x) \setminus f = \emptyset$ ，消息内容为：

$\mu_{x \to f}(x) = 1 \quad \text{(uniform message)}$

### 4.2 Factor-to-Variable Messages
4.2 因子到变量的信息

A factor node $f$ sends a message to a neighboring variable node $x$ by:
因子节点 $f$ 通过以下方式向相邻变量节点 $x$ 发送消息：

1.  Multiplying the factor $f(\mathbf{x}_f)$ with all incoming messages from neighboring variables *except* $x$
    将因子 $f(\mathbf{x}_f)$ 与*除* $x$ 以外的所有相邻变量的传入消息相乘
2.  Summing (marginalizing) over all variables *except* $x$
    对除 $x$ *之外的*所有变量求和（边缘化）

$\mu_{f \to x}(x) = \sum_{\sim x} f(\mathbf{x}_f) \prod_{y \in N(f) \setminus x} \mu_{y \to f}(y)$

where $\sum_{\sim x}$ denotes summation over all variables in $\mathbf{x}_f$ except $x$.
其中 ∑ ∼ 𝑥 ∑ ∼x ​ 表示对 𝑥 中所有变量求和 𝑓 x f ​ 除了 $x$ 。

**Intuition**: The factor $f$ summarizes how all its other connected variables interact through it, and communicates this summary to $x$.
**直觉** ：因子 $f$ 概括了所有其他与其相关的变量如何通过它相互作用，并将此概括传达给 $x$ 。

### 4.3 Belief Computation
4.3 信念计算

After all messages have been exchanged, the **belief** (approximate marginal) at each variable node is computed as the product of all incoming factor-to-variable messages:
所有消息交换完毕后，每个变量节点的**置信度** （近似边缘置信度）计算为所有传入的因子到变量消息的乘积：

$b(x_i) \propto \prod_{f \in N(x_i)} \mu_{f \to x_i}(x_i)$

The belief $b(x_i)$ is then normalized to be a valid probability distribution.
然后，将信念 $b(x_i)$ 归一化为一个有效的概率分布。

* * *

## 5\. The Sum-Product Algorithm
5\. 和积算法

The **Sum-Product algorithm** is the concrete instantiation of Belief Propagation for computing **marginal probabilities**. It is called "Sum-Product" because the factor-to-variable message involves a *sum* (marginalization) of a *product* (factor times incoming messages).
**求和-乘积算法**是置信传播算法计算**边缘概率**的具体实现。之所以称为“求和-乘积”，是因为因子到变量的消息传递涉及*乘积* （因子乘以传入消息）的*求和* （边缘化）。

> *Animation: Sum-Product Algorithm — see `media/videos/bp_scenes/720p30/SumProductAlgorithm.mp4`
> 动画：和积算法 — 参见 `media/videos/bp_scenes/720p30/SumProductAlgorithm.mp4`*

### 5.1 Algorithm on Trees
5.1 树上的算法

On tree-structured factor graphs (no cycles), the Sum-Product algorithm proceeds in two passes:
在树状结构的因子图（无环）上，求和-乘积算法分两步进行：

#### Forward Pass (Leaves to Root)
前向传递（从叶子到根）

1.  Choose an arbitrary root node
    选择一个任意根节点
2.  Starting from the leaf nodes, send messages toward the root
    从叶节点开始，向根节点发送消息。
3.  Each node sends its message only after it has received all incoming messages from its children
    每个节点只有在收到所有来自其子节点的消息后才会发送自己的消息。

#### Backward Pass (Root to Leaves)
反向传递（从根到叶）

4.  The root sends messages back to its children
    根节点会向其子节点发送消息
5.  Messages propagate outward until they reach all leaf nodes
    信息向外传播，直到到达所有叶节点。

#### Marginal Computation
边际计算

6.  At each variable node $x_i$, the marginal is computed as:
    在每个变量节点𝑥 𝑖 x i ​ 边际效应的计算方法如下：

$p(x_i) = \frac{1}{Z_i} \prod_{f \in N(x_i)} \mu_{f \to x_i}(x_i)$

where $Z_i$ is a normalization constant.
其中𝑍 𝑖 Z i ​ 是归一化常数。

### 5.2 Correctness on Trees
5.2 树上的正确性

**Theorem**: On a tree-structured factor graph, the Sum-Product algorithm computes the **exact** marginal distributions for all variables after a single forward-backward pass.
**定理** ：在树状结构的因子图上，求和-乘积算法在一次前向-后向传递后即可计算出所有变量的**精确**边缘分布。

**Proof sketch**: On a tree, every path between two nodes is unique. Therefore, when a node computes its belief using incoming messages, each piece of information (from each factor) is counted exactly once. There is no "double-counting" — the fundamental problem that arises in graphs with cycles.
**证明概要** ：在树状图中，两个节点之间的每条路径都是唯一的。因此，当一个节点使用传入的消息计算其信念时，每个信息（来自每个因子）都恰好被计数一次。不存在“重复计数”——这是带环图中出现的根本问题。

### 5.3 Complexity
5.3 复杂性

For a tree with $n$ variable nodes, where each variable takes at most $k$ values and each factor connects at most $d$ variables:
对于一个具有 $n$ 个变量节点的树，其中每个变量最多取 $k$ 个值，并且每个因子最多连接 $d$ 个变量：

*   **Message computation**: $O(k^d)$ per message (summing over neighbor configurations)
    **消息计算** ：每条消息 $O(k^d)$ （对邻居配置求和）
*   **Total messages**: $O(n)$ (two messages per edge, one in each direction)
    **消息总数** ： $O(n)$ （每条边两条消息，每个方向一条）
*   **Overall complexity**: $O(n \cdot k^d)$, which is linear in the number of variables — a dramatic improvement over the brute-force $O(k^n)$.
    **总体复杂度** ： $O(n \cdot k^d)$ ，与变量数量呈线性关系——比暴力搜索 $O(k^n)$ 有了显著改进。

* * *

## 6\. The Max-Product Algorithm
6\. 最大乘积算法

While the Sum-Product algorithm computes marginal probabilities, many applications require finding the **Maximum A Posteriori (MAP)** assignment — the single most probable configuration of all variables:
虽然求和乘积算法计算的是边缘概率，但许多应用需要找到**最大后验概率 (MAP)** 分配——所有变量的最可能配置：

$\mathbf{x}^* = \arg\max_{\mathbf{x}} p(\mathbf{x}) = \arg\max_{\mathbf{x}} \prod_a f_a(\mathbf{x}_a)$

The **Max-Product algorithm** achieves this by replacing the summation in the factor-to-variable message with a maximization:
**最大乘积算法**通过将因子到变量消息中的求和替换为最大化来实现这一点：

![Sum-Product vs Max-Product Comparison](media/images/bp_scenes/MaxProductComparison_ManimCE_v0.19.2.png)

### 6.1 Message Update Rules
6.1 消息更新规则

**Variable-to-Factor** (same as Sum-Product):
**变量到因子** （与和-积相同）：

$\mu_{x \to f}(x) = \prod_{g \in N(x) \setminus f} \mu_{g \to x}(x)$

**Factor-to-Variable** (max replaces sum):
**因子到变量的转换** （最大值代替总和）：

$\mu_{f \to x}(x) = \max_{\sim x} \left[ f(\mathbf{x}_f) \prod_{y \in N(f) \setminus x} \mu_{y \to f}(y) \right]$

### 6.2 MAP Estimation
6.2 MAP 估计

After convergence, the MAP estimate at each variable is:
收敛后，每个变量的最大后验概率估计值为：

$x_i^* = \arg\max_{x_i} \prod_{f \in N(x_i)} \mu_{f \to x_i}(x_i)$

### 6.3 Min-Sum (Log-Domain) Variant
6.3 最小和（对数域）变体

In practice, it is often more numerically stable to work in the **log domain**. Taking the negative logarithm transforms the Max-Product algorithm into the **Min-Sum** algorithm:
实际上，在对**数域中**进行运算通常数值稳定性更高。取负对数可以将最大乘积算法转化为**最小和**算法：

*   Products become sums: $\log(a \cdot b) = \log a + \log b$
    乘积变为和： $\log(a \cdot b) = \log a + \log b$
*   Maximization becomes minimization (with negation): $\max \to \min$
    最大化变为最小化（带否定）： $\max \to \min$

This avoids numerical underflow issues that arise when multiplying many small probabilities, and is closely related to the **Viterbi algorithm** for finding the most likely sequence in Hidden Markov Models.
这样就避免了将许多小概率相乘时出现的数值下溢问题，并且与用于在隐马尔可夫模型中寻找最可能序列的**维特比算法**密切相关。

### 6.4 Correctness
6.4 正确性

On tree-structured graphs, the Max-Product algorithm finds the **exact** MAP assignment. On loopy graphs, it provides an approximation.
对于树状图，最大乘积算法可以找到**精确的**最大后验概率 (MAP) 分配。对于环状图，它只能提供近似值。

* * *

## 7\. Exact Inference on Trees
7\. 树上的精确推理

Tree-structured factor graphs are special because Belief Propagation yields **exact** results. This section formalizes why this is the case and describes the two-pass message schedule.
树状结构的因子图之所以特殊，是因为置信传播能够产生**精确**结果。本节将阐述其原因，并描述两遍消息调度。

> *Animation: Exact Inference on Trees — see `media/videos/bp_scenes/720p30/TreeBP.mp4`
> 动画：树上的精确推理 — 参见 `media/videos/bp_scenes/720p30/TreeBP.mp4`*

### 7.1 Two-Pass Message Schedule
7.1 两遍消息调度

Given a tree-structured factor graph:
给定一个树状结构的因子图：

**Pass 1 (Leaves → Root):
第一阶段（叶→根）：**

1.  Select any variable node as the root
    选择任意变量节点作为根节点
2.  All leaf nodes send their messages (uniform for variables, or the factor value for factor leaves)
    所有叶子节点发送消息（变量叶子节点发送统一消息，因子叶子节点发送因子值消息）。
3.  Each non-leaf node waits to receive messages from all children, then sends a single message to its parent
    每个非叶子节点等待接收来自所有子节点的消息，然后向其父节点发送一条消息。
4.  This continues until the root has received messages from all children
    这个过程会一直持续到根节点收到所有子节点的消息为止。

**Pass 2 (Root → Leaves):** 5. The root sends messages to all its children 6. Each node, upon receiving a message from its parent, sends messages to all its children 7. This continues until all leaf nodes have received messages
**第二阶段（根节点→叶节点）：** 5. 根节点向其所有子节点发送消息。6. 每个节点在收到其父节点的消息后，向其所有子节点发送消息。7. 此过程持续进行，直到所有叶节点都收到消息为止。

After both passes, every edge has carried exactly two messages (one in each direction), and every node can compute its exact marginal.
经过两次传递后，每条边都恰好承载了两条消息（每个方向一条），并且每个节点都可以计算其精确的边缘分布。

### 7.2 Why Trees Are Special
7.2 为什么树木如此特别

The key property of trees is that they contain **no cycles**. This means:
树的关键特性是它们**不包含环路** 。这意味着：

1.  **Unique paths**: There is exactly one path between any two nodes
    **唯一路径** ：任意两个节点之间都只有一条路径。
2.  **No double-counting**: Each factor contributes to a variable's belief exactly once
    **不重复计算** ：每个因素对变量的置信度仅贡献一次。
3.  **Convergence in finite steps**: The two-pass schedule terminates after visiting each edge twice
    **有限步收敛** ：两遍调度算法在访问每条边两次后终止。

In contrast, graphs with cycles can cause messages to reinforce themselves, leading to the double-counting of evidence — the fundamental challenge addressed by Loopy BP.
相比之下，带有循环的图会导致信息自我强化，从而导致证据重复计算——这是 Loopy BP 所解决的根本挑战。

### 7.3 Junction Tree Algorithm
7.3 连接树算法

For general graphs, exact inference can still be performed by converting the graph into a **junction tree** (also called a clique tree) through a process of triangulation and clique identification. The Sum-Product algorithm on the junction tree yields exact marginals. However, the complexity depends on the **treewidth** of the graph — for graphs with large treewidth, this approach becomes intractable, motivating approximate methods like Loopy BP.
对于一般图，可以通过三角剖分和团识别过程将图转换为**连接树** （也称为团树），从而实现精确推理。连接树上的和积算法可以得到精确的边缘分布。然而，其复杂度取决于图的**树宽** ——对于树宽较大的图，这种方法变得难以处理，因此需要使用诸如 Loopy BP 之类的近似方法。

* * *

## 8\. Loopy Belief Propagation
8\. 循环信念传播

When the factor graph contains **cycles** (loops), the standard two-pass schedule cannot be applied, and messages may travel around loops indefinitely. **Loopy Belief Propagation** (LBP) applies the same message update rules iteratively until (approximate) convergence.
当因子图包含**环路**时，标准的两遍调度方法无法应用，消息可能会无限循环地在环路中传播。 **循环置信传播** （LBP）会迭代地应用相同的消息更新规则，直到（近似）收敛。

> *Animation: Loopy BP on a Cyclic Graph — see `media/videos/bp_scenes/720p30/LoopyBP.mp4`
> 动画：循环图上的循环 BP — 参见 `media/videos/bp_scenes/720p30/LoopyBP.mp4`*

### 8.1 Algorithm
8.1 算法

1.  **Initialize** all messages to uniform distributions (or random)
    将所有消息**初始化**为均匀分布（或随机分布）。
2.  **Iterate**: For each edge in the graph, update the messages using the Sum-Product (or Max-Product) update rules
    **迭代** ：对于图中的每条边，使用求和-乘积（或最大乘积）更新规则更新消息。
3.  **Repeat** until messages change by less than a threshold $\epsilon$, or a maximum number of iterations is reached
    **重复此过程** ，直到消息变化小于阈值 $\epsilon$ ，或达到最大迭代次数为止。
4.  **Compute beliefs** from the final messages
    根据最终消息**计算置信度**

### 8.2 Message Schedule
8.2 消息计划

Several scheduling strategies exist:
存在多种排课策略：

| Schedule日程 | Description描述 |
| --- | --- |
| Synchronous (Flooding)同步（泛洪） | All messages updated simultaneously in each iteration每次迭代中所有消息同时更新 |
| Asynchronous (Sequential)异步（顺序） | Messages updated one at a time in some order消息按某种顺序逐条更新。 |
| Residual BP残余血压 | Prioritize updating messages with largest residual (change)优先更新变化量最大的消息 |

### 8.3 Convergence Properties
8.3 收敛性质

Unlike the tree case, Loopy BP has **no general convergence guarantee**:
与树形问题不同，Loopy BP **没有一般的收敛性保证** ：

*   On some graphs, messages converge to a fixed point that provides excellent marginal approximations
    在某些图上，消息会收敛到一个固定点，该固定点提供了极佳的边缘近似值。
*   On others, messages may **oscillate** or even **diverge**
    在其他情况下，信息可能会**波动**甚至**出现分歧。**
*   Convergence is more likely when:
    当出现以下情况时，趋同的可能性更大：
    *   The graph has long loops (weak interactions around cycles)
        该图存在长环（环周围的相互作用较弱）。
    *   The factors/potentials are "weak" (close to uniform)
        这些因素/潜力“较弱”（接近均匀分布）。
    *   **Damping** is applied: $\mu^{(t+1)} = \alpha \cdot \mu^{\text{new}} + (1-\alpha) \cdot \mu^{(t)}$
        应用**阻尼** ： $\mu^{(t+1)} = \alpha \cdot \mu^{\text{new}} + (1-\alpha) \cdot \mu^{(t)}$

### 8.4 Theoretical Foundations
8.4 理论基础

When Loopy BP converges, the fixed point can be characterized as a stationary point of the **Bethe free energy**:
当 Loopy BP 收敛时，不动点可以被描述为 **Bethe 自由能**的驻点：

$F_{\text{Bethe}} = \sum_a \sum_{\mathbf{x}_a} b_a(\mathbf{x}_a) \left[ \ln b_a(\mathbf{x}_a) - \ln f_a(\mathbf{x}_a) \right] - \sum_i (d_i - 1) \sum_{x_i} b_i(x_i) \ln b_i(x_i)$

where $b_a$ and $b_i$ are the factor and variable beliefs, and $d_i$ is the degree of variable node $i$. This connection to variational inference (Yedidia, Freeman, and Weiss, 2001) provides theoretical justification for Loopy BP and has led to improved variants.
其中 𝑏 𝑎 b a ​ 和 𝑏 𝑖 b i ​ 是因素和变量信念，以及 𝑑 𝑖 d i ​ 是变量节点 $i$ 的度。这种与变分推断（Yedidia、Freeman 和 Weiss，2001）的联系为 Loopy BP 提供了理论依据，并导致了改进的变体。

### 8.5 Practical Considerations
8.5 实际考虑因素

Despite lacking formal guarantees, Loopy BP is remarkably effective in practice:
尽管缺乏正式的保证，Loopy BP 在实践中却非常有效：

*   **Turbo codes** and **LDPC codes**: Near-Shannon-limit performance in decoding
    **Turbo 码**和 **LDPC 码** ：译码性能接近香农极限
*   **Stereo vision**: State-of-the-art depth estimation
    **立体视觉** ：最先进的深度估计
*   **Protein folding**: Prediction of molecular structures
    **蛋白质折叠** ：分子结构预测

The empirical success of Loopy BP, combined with its simplicity and efficiency, makes it one of the most important algorithms in probabilistic inference.
Loopy BP 的经验成功，加上其简单性和高效性，使其成为概率推理中最重要的算法之一。

* * *

## 9\. Numerical Example
9\. 数值示例

To make the algorithm concrete, consider a simple chain factor graph with three binary variables:
为了使算法具体化，考虑一个具有三个二元变量的简单链因子图：

$p(x_1, x_2, x_3) = f_1(x_1, x_2) \cdot f_2(x_2, x_3)$

where each variable takes values in $\{0, 1\}$, and the factor tables are:
其中每个变量的取值范围为 $\{0, 1\}$ ，因子表如下：

$f_1(x_1, x_2) = \begin{pmatrix} 0.8 & 0.2 \\ 0.3 & 0.7 \end{pmatrix}, \quad f_2(x_2, x_3) = \begin{pmatrix} 0.6 & 0.4 \\ 0.1 & 0.9 \end{pmatrix}$

Here, $f_1(x_1=0, x_2=0) = 0.8$, $f_1(x_1=0, x_2=1) = 0.2$, etc.
这里， $f_1(x_1=0, x_2=0) = 0.8$ ， $f_1(x_1=0, x_2=1) = 0.2$ ，等等。

> *Animation: Belief Convergence — see `media/videos/bp_scenes/720p30/BeliefConvergence.mp4`
> 动画：信念收敛 — 参见 `media/videos/bp_scenes/720p30/BeliefConvergence.mp4`*

### 9.1 Step-by-Step Computation
9.1 逐步计算

**Initialization**: All messages set to $(1, 1)$ (uniform).
**初始化** ：所有消息设置为 $(1, 1)$ （统一）。

**Forward Pass (left to right):
前传（从左到右）：**

1.  **Leaf message** $\mu_{x_1 \to f_1}(x_1) = (1, 1)$
    **叶子消息** $\mu_{x_1 \to f_1}(x_1) = (1, 1)$
    
2.  **Factor-to-variable** $\mu_{f_1 \to x_2}(x_2)$:
    **因子到变量** $\mu_{f_1 \to x_2}(x_2)$ ：
    

$\mu_{f_1 \to x_2}(x_2=0) = \sum_{x_1} f_1(x_1, x_2=0) \cdot \mu_{x_1 \to f_1}(x_1) = 0.8 + 0.3 = 1.1$

$\mu_{f_1 \to x_2}(x_2=1) = \sum_{x_1} f_1(x_1, x_2=1) \cdot \mu_{x_1 \to f_1}(x_1) = 0.2 + 0.7 = 0.9$

After normalization: $\mu_{f_1 \to x_2} = (0.55, 0.45)$
归一化后： $\mu_{f_1 \to x_2} = (0.55, 0.45)$

3.  **Variable-to-factor** $\mu_{x_2 \to f_2} = \mu_{f_1 \to x_2} = (0.55, 0.45)$
    **变量到因子** $\mu_{x_2 \to f_2} = \mu_{f_1 \to x_2} = (0.55, 0.45)$
    
4.  **Factor-to-variable** $\mu_{f_2 \to x_3}(x_3)$:
    **因子到变量** $\mu_{f_2 \to x_3}(x_3)$ ：
    

$\mu_{f_2 \to x_3}(x_3=0) = 0.55 \times 0.6 + 0.45 \times 0.1 = 0.375$

$\mu_{f_2 \to x_3}(x_3=1) = 0.55 \times 0.4 + 0.45 \times 0.9 = 0.625$

After normalization: $\mu_{f_2 \to x_3} = (0.375, 0.625)$
归一化后： $\mu_{f_2 \to x_3} = (0.375, 0.625)$

**Backward Pass (right to left):
后传（从右到左）：**

5.  **Leaf message** $\mu_{x_3 \to f_2}(x_3) = (1, 1)$
    **叶子消息** $\mu_{x_3 \to f_2}(x_3) = (1, 1)$
    
6.  **Factor-to-variable** $\mu_{f_2 \to x_2}(x_2)$:
    **因子到变量** $\mu_{f_2 \to x_2}(x_2)$ ：
    

$\mu_{f_2 \to x_2}(x_2=0) = 0.6 + 0.4 = 1.0$

$\mu_{f_2 \to x_2}(x_2=1) = 0.1 + 0.9 = 1.0$

After normalization: $\mu_{f_2 \to x_2} = (0.5, 0.5)$
归一化后： $\mu_{f_2 \to x_2} = (0.5, 0.5)$

**Belief Computation:
信念计算：**

$b(x_2) \propto \mu_{f_1 \to x_2} \cdot \mu_{f_2 \to x_2} = (0.55 \times 0.5, 0.45 \times 0.5) = (0.275, 0.225)$

After normalization: $b(x_2) = (0.55, 0.45)$
归一化后： $b(x_2) = (0.55, 0.45)$

The animation below shows how the belief distributions for all three variables converge over iterations of the BP algorithm.
下面的动画展示了 BP 算法迭代过程中所有三个变量的置信分布是如何收敛的。

* * *

## 10\. Applications
10\. 应用

Belief Propagation and its variants have found widespread use across numerous domains:
信念传播及其变体已在众多领域得到广泛应用：

### 10.1 Error-Correcting Codes
10.1 纠错码

BP is the decoding algorithm underlying two of the most important classes of error-correcting codes:
BP 是两种最重要的纠错码的基础解码算法：

*   **Turbo Codes** (Berrou et al., 1993): Use iterative BP-like decoding between two constituent convolutional codes, achieving near-Shannon-limit performance.
    **Turbo 码** （Berrou 等人，1993 年）：在两个组成卷积码之间使用迭代 BP 类解码，达到接近香农极限的性能。
*   **Low-Density Parity-Check (LDPC) Codes** (Gallager, 1962; rediscovered by MacKay, 1999): The parity-check matrix defines a factor graph, and BP decoding (also called "sum-product decoding") achieves remarkable performance. LDPC codes are used in 5G, Wi-Fi (802.11n/ac), and digital television standards.
    **低密度奇偶校验码（LDPC）** （Gallager，1962；MacKay，1999 年重新发现）：奇偶校验矩阵定义了一个因子图，BP 译码（也称为“和积译码”）可实现卓越的性能。LDPC 码应用于 5G、Wi-Fi（802.11n/ac）和数字电视标准中。

### 10.2 Computer Vision
10.2 计算机视觉

*   **Stereo Matching**: BP on MRFs finds pixel-wise depth maps by enforcing smoothness between neighboring pixels while matching left-right image pairs.
    **立体匹配** ：BP on MRFs 通过强制相邻像素之间的平滑性来匹配左右图像对，从而找到像素级深度图。
*   **Image Segmentation**: MRF-based models with BP inference assign semantic labels to image regions.
    **图像分割** ：基于 MRF 的 BP 推理模型为图像区域分配语义标签。
*   **Object Detection and Pose Estimation**: Pictorial structure models use BP to efficiently reason about spatial configurations of object parts.
    **目标检测和姿态估计** ：图像结构模型使用 BP 来有效地推理物体部分的空间配置。
*   **Image Denoising**: Removing noise from images by propagating local evidence through an MRF.
    **图像去噪** ：通过 MRF 传播局部证据来去除图像中的噪声。

### 10.3 Natural Language Processing
10.3 自然语言处理

*   **Part-of-Speech Tagging**: The forward-backward algorithm (a special case of BP on HMMs) computes marginal tag probabilities.
    **词性标注** ：前向-后向算法（HMM 上的 BP 算法的一个特例）计算边缘标签概率。
*   **Named Entity Recognition**: CRF models decoded with BP.
    **命名实体识别** ：使用 BP 解码的 CRF 模型。
*   **Machine Translation**: Alignment models and syntax-based translation use BP-like message passing.
    **机器翻译** ：对齐模型和基于语法的翻译使用类似 BP 的消息传递。

### 10.4 Computational Biology
10.4 计算生物学

*   **Protein Structure Prediction**: BP on residue interaction networks.
    **蛋白质结构预测** ：基于残基相互作用网络的 BP。
*   **Gene Regulatory Networks**: Inferring gene expression states.
    **基因调控网络** ：推断基因表达状态。
*   **Phylogenetics**: Computing likelihoods on evolutionary trees (Felsenstein's pruning algorithm is a special case of BP).
    **系统发育学** ：计算进化树的似然性（Felsenstein 剪枝算法是 BP 的一个特例）。

### 10.5 Robotics and SLAM
10.5 机器人与 SLAM

*   **Simultaneous Localization and Mapping (SLAM)**: Factor graph models with BP solve the robot localization problem.
    **同时定位与建图（SLAM）** ：采用 BP 算法的因子图模型解决机器人定位问题。
*   **Sensor Fusion**: Combining information from multiple noisy sensors using message passing.
    **传感器融合** ：利用消息传递将来自多个噪声传感器的信息结合起来。

* * *

## 11\. Conclusion
11\. 结论

Belief Propagation is a powerful and versatile algorithm for probabilistic inference on graphical models. Its key strengths include:
信念传播算法是一种功能强大且用途广泛的算法，适用于图模型上的概率推理。其主要优势包括：

1.  **Exactness on trees**: The Sum-Product algorithm provably computes exact marginals on tree-structured factor graphs in linear time.
    **树上的精确性** ：Sum-Product 算法可证明能在线性时间内计算树状因子图上的精确边缘分布。
2.  **Practical effectiveness on loopy graphs**: Despite lacking convergence guarantees, Loopy BP provides excellent approximations in many real-world applications.
    **在循环图上的实际有效性** ：尽管缺乏收敛性保证，但 Loopy BP 在许多实际应用中提供了极佳的近似结果。
3.  **Modularity**: The local message-passing rules are simple and modular — the algorithm naturally decomposes global inference into local computations.
    **模块化** ：局部消息传递规则简单且模块化——该算法自然地将全局推理分解为局部计算。
4.  **Versatility**: By changing the "sum" to a "max", the same framework handles both marginal inference (Sum-Product) and MAP inference (Max-Product).
    **多功能性** ：通过将“求和”改为“最大值”，同一个框架可以处理边际推断（求和-乘积）和最大后验概率推断（最大-乘积）。
5.  **Theoretical depth**: Connections to variational inference, the Bethe free energy, and information geometry provide a rich theoretical understanding.
    **理论深度** ：与变分推断、贝特自由能和信息几何的联系提供了丰富的理论理解。

Future directions include:
未来发展方向包括：

*   **Neural Belief Propagation**: Combining BP with neural networks for learned message functions
    **神经信念传播** ：将反向传播算法与神经网络相结合，用于学习消息函数
*   **Generalized BP**: Extensions like Expectation Propagation and Region-based BP that improve approximation quality
    **广义 BP** ：诸如期望传播和基于区域的 BP 等扩展方法可以提高逼近质量
*   **Quantum Belief Propagation**: Adapting message passing for quantum probabilistic models
    **量子信念传播** ：将消息传递应用于量子概率模型

Belief Propagation remains one of the most elegant and practically impactful algorithms at the intersection of probability theory, graph theory, and computer science.
信念传播仍然是概率论、图论和计算机科学交叉领域中最优雅、最具实际影响力的算法之一。

* * *

## 12\. References
12\. 参考文献

1.  Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.
    Pearl, J. (1988). *智能系统中的概率推理：似然推理网络* 。Morgan Kaufmann 出版社。
    
2.  Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor graphs and the sum-product algorithm. *IEEE Transactions on Information Theory*, 47(2), 498–519.
    Kschischang, FR, Frey, BJ, & Loeliger, H.-A. (2001). 因子图和求和乘积算法。IEEE *信息论汇刊* ，47(2), 498–519。
    
3.  Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2001). Understanding belief propagation and its generalizations. *Exploring Artificial Intelligence in the New Millennium*, 8, 236–239.
    Yedidia, JS, Freeman, WT, & Weiss, Y. (2001). 理解信念传播及其概括。 *探索新千年的人工智能* ，8，236–239。
    
4.  Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. (Chapter 8: Graphical Models)
    Bishop, CM (2006). *模式识别与机器学习* . Springer. (第 8 章：图形模型)
    
5.  Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
    Koller, D. 和 Friedman, N. (2009)。 *概率图模型：原理与技术* 。麻省理工学院出版社。
    
6.  Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. (Chapter 20: Exact Inference for Graphical Models; Chapter 22: Variational Inference)
    Murphy, KP (2012). *机器学习：概率视角* 。麻省理工学院出版社。（第 20 章：图模型的精确推理；第 22 章：变分推理）
    
7.  Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. *Foundations and Trends in Machine Learning*, 1(1–2), 1–305.
    Wainwright, MJ 和 Jordan, MI (2008)。图形模型、指数族和变分推断。 *机器学习基础与趋势* ，1(1–2)，1–305。
    
8.  Berrou, C., Glavieux, A., & Thitimajshima, P. (1993). Near Shannon limit error-correcting coding and decoding: Turbo-codes. *Proceedings of IEEE ICC*, 1064–1070.
    Berrou, C., Glavieux, A., & Thitimajshima, P. (1993). 接近香农极限的纠错编码和解码：Turbo 码。IEEE *ICC 会议论文集* ，1064–1070。
    
9.  MacKay, D. J. C. (1999). Good error-correcting codes based on very sparse matrices. *IEEE Transactions on Information Theory*, 45(2), 399–431.
    MacKay, DJC (1999). 基于稀疏矩阵的良好纠错码. *IEEE 信息论汇刊* , 45(2), 399–431.
    
10.  Felzenszwalb, P. F., & Huttenlocher, D. P. (2006). Efficient belief propagation for early vision. *International Journal of Computer Vision*, 70(1), 41–54.
    Felzenszwalb, PF, & Huttenlocher, DP (2006). 早期视觉的有效信念传播. *国际计算机视觉杂志* , 70(1), 41–54。