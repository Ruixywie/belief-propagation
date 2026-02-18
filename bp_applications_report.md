# 置信传播（BP）算法的应用报告 | Application Report on Belief Propagation (BP) Algorithm

## 引言 | Introduction

置信传播（Belief Propagation, BP）是一种在图模型上进行概率推理的消息传递算法，由 Judea Pearl 于 1982 年提出。其核心思想是：图中每个节点通过与邻居交换"消息"来更新自身对变量取值的"置信度"，最终收敛到（近似）后验概率。在树结构图上 BP 可以精确求解，在含环图（Loopy BP）上则作为高效的近似推理方法被广泛应用。

Belief Propagation (BP) is a message-passing algorithm for probabilistic inference on graphical models, proposed by Judea Pearl in 1982. Its core idea is that each node in the graph exchanges "messages" with its neighbors to update its "belief" about variable assignments, eventually converging to (approximate) posterior probabilities. On tree-structured graphs, BP yields exact solutions; on graphs with loops (Loopy BP), it serves as an efficient approximate inference method and has been widely adopted.

本报告聚焦 BP 算法在三个不同领域中的具体应用：**图像降噪**、**立体匹配**和**蛋白质侧链构象预测**。每个模块以实际案例为核心，深入分析算法建模、效果表现、BP 的核心作用、现存不足，以及与当前前沿方法的差距。

This report focuses on the application of BP in three distinct domains: **image denoising**, **stereo matching**, and **protein side-chain conformation prediction**. Each module is centered on a concrete case study, providing in-depth analysis of algorithmic modeling, performance, the role of BP, existing limitations, and the gap compared to state-of-the-art methods.

---

## 模块一：基于 BP 的图像降噪 | Module 1: BP-Based Image Denoising

> **技术总结 | Technical Summary**
>
> **算法**：Min-Sum Loopy BP（对数域 Max-Product BP）| **Algorithm**: Min-Sum Loopy BP (log-domain Max-Product BP)
> **图模型**：4-连接网格 MRF，像素为变量节点 | **Graphical Model**: 4-connected grid MRF with pixels as variable nodes
> **一元势**：高斯似然 $(v(l) - y)^2 / 2\sigma^2$ | **Unary**: Gaussian likelihood $(v(l) - y)^2 / 2\sigma^2$
> **二元势**：截断二次/线性模型（边缘保留）| **Pairwise**: Truncated quadratic/linear model (edge-preserving)
> **关键技术**：距离变换加速 + 阻尼消息更新 + float32 优化 | **Key Techniques**: Distance transform acceleration + damped message updates + float32 optimization
> **推理目标**：MAP 估计（能量最小化）| **Inference Goal**: MAP estimation (energy minimization)

### 案例：本组 Web 降噪应用 | Case Study: Our Web Denoising Application

本模块的案例来自本组实现的交互式图像降噪 Web 应用（`Group/application/`），采用 Flask 框架，后端核心为 Min-Sum Loopy BP 算法在 4-连接 MRF 网格上的实现。

The case study for this module is our team's interactive image denoising web application (`Group/application/`), built with the Flask framework. The backend core implements the Min-Sum Loopy BP algorithm on a 4-connected MRF grid.

### 1.1 问题建模 | Problem Formulation

图像降噪的目标是从含噪观测图像 $y$ 中恢复出洁净图像 $x$。我们将灰度图像建模为**马尔可夫随机场（MRF）**：

The goal of image denoising is to recover a clean image $x$ from a noisy observation $y$. We model the grayscale image as a **Markov Random Field (MRF)**:

- **变量节点 (Variable nodes)**：每个像素位置 $(i,j)$ 对应一个变量 $x_{ij}$，取值为量化后的离散标签 $l \in \{0, 1, \dots, L-1\}$
  Each pixel location $(i,j)$ corresponds to a variable $x_{ij}$, taking a quantized discrete label $l \in \{0, 1, \dots, L-1\}$.
- **因子节点 (Factor nodes)**：包含两类势函数 | Two types of potential functions:
  - **一元势（数据项）| Unary potential (data term)**：编码观测数据 $y_{ij}$ 对变量 $x_{ij}$ 的约束 | Encodes the constraint of observed data $y_{ij}$ on variable $x_{ij}$
  - **二元势（平滑项）| Pairwise potential (smoothness term)**：编码相邻像素间的空间平滑先验 | Encodes spatial smoothness priors between neighboring pixels
- **图结构 (Graph structure)**：4-连接网格——每个像素与上、下、左、右四个邻居相连
  4-connected grid — each pixel is connected to its four neighbors (up, down, left, right).

整个问题归结为在 MRF 上求解**最大后验（MAP）估计**：

The problem reduces to solving **Maximum A Posteriori (MAP) estimation** on the MRF:

$$\hat{x} = \arg\min_x \sum_{i,j} \phi(x_{ij}, y_{ij}) + \sum_{(i,j) \sim (i',j')} \psi(x_{ij}, x_{i'j'})$$

其中 $\phi$ 为一元代价（数据保真项），$\psi$ 为二元代价（平滑正则项），$(i,j) \sim (i',j')$ 表示像素 $(i,j)$ 和 $(i',j')$ 为 4-连通邻居。

where $\phi$ is the unary cost (data fidelity term), $\psi$ is the pairwise cost (smoothness regularization term), and $(i,j) \sim (i',j')$ denotes that pixels $(i,j)$ and $(i',j')$ are 4-connected neighbors.

### 1.2 算法细节 | Algorithm Details

#### 标签量化 | Label Quantization

将连续像素值 $[0, 255]$ 均匀离散化为 $L$ 个标签（默认 $L = 64$），标签 $l$ 对应的像素值为：

Continuous pixel values $[0, 255]$ are uniformly discretized into $L$ labels (default $L = 64$). The pixel value corresponding to label $l$ is:

$$v(l) = l \times \frac{256}{L} + \frac{256}{2L}$$

#### 一元势函数（数据保真项）| Unary Potential (Data Fidelity Term)

采用高斯似然代价：

A Gaussian likelihood cost is used:

$$\phi(x_{ij} = l,\ y_{ij}) = \frac{(v(l) - y_{ij})^2}{2\sigma^2}$$

其中 $\sigma$ 为噪声标准差。该项惩罚重建值偏离观测值的程度。

where $\sigma$ is the noise standard deviation. This term penalizes the deviation of the reconstructed value from the observation.

#### 二元势函数（平滑正则项）| Pairwise Potential (Smoothness Regularization)

提供两种截断模型，通过参数 `model` 选择：

Two truncated models are available, selected by the `model` parameter:

**截断线性模型 | Truncated linear model** (`model='linear'`):

$$\psi(x_{ij}, x_{i'j'}) = \min\bigl(d_{\text{disc}},\ w \cdot |l_{ij} - l_{i'j'}|\bigr)$$

**截断二次模型 | Truncated quadratic model** (`model='quadratic'`, default):

$$\psi(x_{ij}, x_{i'j'}) = \min\bigl(d_{\text{disc}},\ w \cdot (l_{ij} - l_{i'j'})^2\bigr)$$

其中 $w$ 为平滑权重（自动缩放为 $w = \text{smooth\_weight} \times \frac{\text{step}^2}{2\sigma^2}$），$d_{\text{disc}}$ 为截断阈值（对应边缘保留——标签差异超过阈值的像素对不再被额外惩罚，从而保留边缘）。截断二次模型对小差异惩罚较温和、对大差异惩罚较强，具有更好的梯度保持特性。

where $w$ is the smoothing weight (auto-scaled as $w = \text{smooth\_weight} \times \frac{\text{step}^2}{2\sigma^2}$), and $d_{\text{disc}}$ is the truncation threshold (for edge preservation — pixel pairs with label differences beyond the threshold receive no additional penalty, thus preserving edges). The truncated quadratic model penalizes small differences gently and large differences heavily, offering better gradient preservation.

#### Min-Sum 消息传递 | Min-Sum Message Passing

算法采用 **Min-Sum**（即对数域下的 Max-Product）进行 MAP 推理。每次迭代中，四个方向（上→下、下→上、左→右、右→左）依次更新消息：

The algorithm uses **Min-Sum** (i.e., log-domain Max-Product) for MAP inference. In each iteration, messages are updated sequentially in four directions (up→down, down→up, left→right, right→left):

1. **聚合 | Aggregation**：对于方向 $d$ 的消息更新，先将一元代价与除反方向 $\bar{d}$ 外的三个方向传入消息求和
   For message update in direction $d$, the unary cost and incoming messages from the three directions (excluding the opposite direction $\bar{d}$) are summed.
2. **距离变换加速 | Distance transform acceleration**：
   - 线性模型使用 $O(L)$ 的前向-后向扫描算法 | Linear model uses $O(L)$ forward-backward sweep
   - 二次模型使用 $O(L^2)$ 的标签循环（预计算代价矩阵）| Quadratic model uses $O(L^2)$ label loop (pre-computed cost matrix)
3. **归一化 | Normalization**：消息减去最小值，防止数值溢出 | Messages are shifted by subtracting the minimum to prevent numerical overflow
4. **移位 | Shifting**：消息沿传播方向移位一个像素 | Messages are shifted by one pixel along the propagation direction
5. **阻尼 | Damping**：新消息 = $\alpha \cdot \text{旧消息} + (1-\alpha) \cdot \text{新消息}$（默认 $\alpha = 0.5$），抑制 Loopy BP 在含环图上的振荡
   New message = $\alpha \cdot \text{old message} + (1-\alpha) \cdot \text{new message}$ (default $\alpha = 0.5$), suppressing oscillations of Loopy BP on graphs with cycles.

#### 置信度计算与 MAP 决策 | Belief Computation and MAP Decision

迭代结束后，计算每个像素的置信度（一元代价 + 四方向消息之和），取 argmin 得到最优标签，再反量化回像素值。

After iterations complete, the belief for each pixel is computed (sum of unary cost and messages from all four directions), and argmin yields the optimal label, which is then dequantized back to a pixel value.

#### 工程优化 | Engineering Optimizations

- 全程使用 `float32` 精度（约 1.5 倍加速，内存减半）
  `float32` precision throughout (~1.5x speedup, half memory usage)
- 每个方向仅拷贝该方向消息（而非全部 4 方向），减少 5 倍内存拷贝
  Per-direction copy instead of copying all 4 directions (5x less memory copy)
- 回调间隔控制：每 $\lceil \text{iters}/15 \rceil$ 次迭代才编码中间结果图像，减少 PIL 编码开销
  Callback interval control: intermediate images encoded only every $\lceil \text{iters}/15 \rceil$ iterations, reducing PIL encoding overhead

### 1.3 效果展示 | Performance Demonstration

在典型参数设置下（$\sigma = 25$，$L = 64$ 标签，30 次迭代，截断二次模型，阻尼 0.5），本应用对标准测试图像的去噪效果如下：

Under typical parameter settings ($\sigma = 25$, $L = 64$ labels, 30 iterations, truncated quadratic model, damping 0.5), the denoising performance on standard test images is as follows:

| 指标 Metric | 典型值 Typical Value |
|-------------|---------------------|
| 含噪图 PSNR / Noisy PSNR | ~20 dB |
| 去噪后 PSNR / Denoised PSNR | ~25–27 dB |
| PSNR 提升 / PSNR Improvement | +5–7 dB |
| 处理时间（200×200）/ Processing time (200×200) | ~tens of seconds |

去噪结果呈现出明显的平滑效果，同时截断机制保留了主要边缘。但在细节纹理区域（如毛发、草地），平滑先验会导致纹理模糊。

The denoised results show significant smoothing while the truncation mechanism preserves major edges. However, in textured regions (e.g., hair, grass), the smoothness prior causes texture blurring.

### 1.4 BP 的作用 | Role of BP

在此应用中，BP 的核心作用是：

In this application, the key roles of BP are:

- **全局 MAP 推理 | Global MAP inference**：通过消息传递，每个像素的决策不仅依赖于自身观测值，还综合了来自图像各处（通过多次迭代扩散）的平滑约束信息
  Through message passing, each pixel's decision depends not only on its own observation but also integrates smoothness constraints propagated from across the image over multiple iterations.
- **平衡数据保真与空间平滑 | Balancing data fidelity and spatial smoothness**：一元势约束重建值接近观测值，二元势约束相邻像素取值一致，BP 在两者之间自动找到平衡点
  The unary potential constrains reconstructed values to stay close to observations, while the pairwise potential enforces consistency between neighboring pixels. BP automatically finds the balance between the two.
- **保边去噪 | Edge-preserving denoising**：截断机制使 BP 能够区分噪声引起的小幅变化（应平滑）和真实边缘引起的大幅变化（应保留）
  The truncation mechanism enables BP to distinguish between small variations caused by noise (to be smoothed) and large variations caused by real edges (to be preserved).

### 1.5 不足与瓶颈 | Limitations and Bottlenecks

1. **计算复杂度高 | High computational complexity**：总复杂度为 $O(H \times W \times L^2 \times \text{iters})$（二次模型）或 $O(H \times W \times L \times \text{iters})$（线性模型）。对于大图像或高标签数，计算时间显著增长。
   Total complexity is $O(H \times W \times L^2 \times \text{iters})$ (quadratic) or $O(H \times W \times L \times \text{iters})$ (linear). Computation time grows significantly for large images or high label counts.
2. **无收敛保证 | No convergence guarantee**：在 4-连接网格这样的含环图上，Loopy BP 理论上不保证收敛。虽然阻尼机制在实践中通常有效，但对某些参数组合可能出现消息振荡。
   On loopy graphs such as 4-connected grids, Loopy BP offers no theoretical convergence guarantee. Although damping is usually effective in practice, message oscillation may occur for certain parameter combinations.
3. **仅支持灰度图 | Grayscale only**：当前实现仅处理单通道灰度图像，无法直接处理彩色图像。
   The current implementation only handles single-channel grayscale images and cannot directly process color images.
4. **高度依赖手动调参 | Heavy reliance on manual parameter tuning**：噪声标准差 $\sigma$、平滑权重、截断阈值、标签数量等参数对结果影响显著，不同图像和噪声水平需要不同的参数组合。
   Parameters such as noise std $\sigma$, smoothing weight, truncation threshold, and label count significantly affect results, requiring different combinations for different images and noise levels.
5. **纹理区域表现差 | Poor performance in textured regions**：4-连接 MRF 的平滑先验对纹理丰富的区域（如草地、织物）不友好，容易过度平滑丢失细节。
   The smoothness prior of the 4-connected MRF is unfavorable for texture-rich regions (e.g., grass, fabric), often over-smoothing and losing fine details.

### 1.6 与前沿方法对比 | Comparison with State-of-the-Art

| 方法 Method | 年份 Year | 类型 Type | σ=25 PSNR (Set12) | 速度 Speed |
|-------------|-----------|-----------|-------------------|------------|
| **MRF-BP (ours)** | — | 概率图模型 PGM | ~25–27 dB | ~tens of sec / 200px |
| BM3D | 2007 | 非局部协同滤波 Non-local collaborative filtering | ~28–31 dB | <1s |
| DnCNN | 2017 | CNN | ~29–31 dB | ~10ms (GPU) |
| SwinIR | 2021 | Swin Transformer | ~30–33 dB | ~50ms (GPU) |
| Restormer | 2022 | Transformer | ~30–33 dB | ~100ms (GPU) |

**分析 | Analysis**：

- BP 方法在同等条件下 **PSNR 通常低 2–5 dB**，且处理速度慢数个量级。
  Under comparable conditions, BP methods typically yield **2–5 dB lower PSNR** and are orders of magnitude slower.
- BM3D 利用图像自相似性的非局部信息，在经典方法中已大幅超越 MRF-BP。
  BM3D exploits non-local self-similarity in images and significantly outperforms MRF-BP among classical methods.
- 深度学习方法（DnCNN、SwinIR、Restormer）通过大规模数据驱动训练，学习到了远超人工设计先验的去噪能力，且推理速度极快。
  Deep learning methods (DnCNN, SwinIR, Restormer), through large-scale data-driven training, learn denoising capabilities far beyond hand-crafted priors, with extremely fast inference.
- BP 方法的优势在于**可解释性强**（每一步都有明确的概率含义）和**不依赖训练数据**。
  The advantages of BP lie in its **strong interpretability** (every step has a clear probabilistic meaning) and **independence from training data**.

---

## 模块二：基于 BP 的立体匹配 | Module 2: BP-Based Stereo Matching

> **技术总结 | Technical Summary**
>
> **算法**：Hierarchical BP（分层置信传播）— 多尺度粗到细 Min-Sum BP | **Algorithm**: Hierarchical BP — multi-scale coarse-to-fine Min-Sum BP
> **图模型**：4-连接网格 MRF，像素为变量节点，视差为标签 | **Graphical Model**: 4-connected grid MRF with pixels as variable nodes, disparity as labels
> **数据代价**：Census Transform 汉明距离 / SAD 窗口匹配 | **Data Cost**: Census Transform Hamming distance / SAD window matching
> **平滑代价**：截断二次距离变换 | **Smoothness Cost**: Truncated quadratic distance transform
> **关键技术**：高斯金字塔 + 消息上采样（标签维线性插值）+ 阻尼 + 后处理（中值/双边滤波）+ CuPy GPU 加速 | **Key Techniques**: Gaussian pyramid + message upsampling (label-dim linear interpolation) + damping + post-processing (median/bilateral filter) + CuPy GPU acceleration
> **推理目标**：MAP 估计（MRF 能量最小化 → 最优视差图）| **Inference Goal**: MAP estimation (MRF energy minimization → optimal disparity map)

### 案例：本组 Web 立体匹配应用 | Case Study: Our Web Stereo Matching Application

本模块的案例来自本组实现的交互式立体匹配 Web 应用（`gp/`），采用 FastAPI 框架，后端核心为分层置信传播（Hierarchical BP）算法，支持 CuPy GPU 加速。

The case study for this module is our team's interactive stereo matching web application (`gp/`), built with the FastAPI framework. The backend core implements Hierarchical Belief Propagation with CuPy GPU acceleration support.

### 2.1 问题建模 | Problem Formulation

立体匹配的目标是：给定经过校正的左右双目图像，估计每个像素的**视差（disparity）**，从而获得场景深度信息。将视差估计建模为 **MRF 能量最小化**问题：

The goal of stereo matching is: given a rectified pair of left and right images, estimate the **disparity** of each pixel to obtain scene depth information. Disparity estimation is formulated as an **MRF energy minimization** problem:

$$E(d) = \sum_{p} C_{\text{data}}(p, d_p) + \lambda \sum_{(p,q) \in \mathcal{N}} V(d_p - d_q)$$

其中：| where:

- $d_p$ 为像素 $p$ 的视差取值，$d_p \in \{0, 1, \dots, D_{\max}-1\}$
  $d_p$ is the disparity of pixel $p$, $d_p \in \{0, 1, \dots, D_{\max}-1\}$
- $C_{\text{data}}(p, d_p)$ 为**数据代价**——左右图像在视差 $d_p$ 下的匹配代价
  $C_{\text{data}}(p, d_p)$ is the **data cost** — matching cost between left and right images at disparity $d_p$
- $V(d_p - d_q)$ 为**平滑代价**——惩罚相邻像素间的视差不一致
  $V(d_p - d_q)$ is the **smoothness cost** — penalizing disparity inconsistency between neighboring pixels
- $\lambda$ 为平滑权重，$\mathcal{N}$ 为 4-连通邻域
  $\lambda$ is the smoothing weight, $\mathcal{N}$ is the 4-connected neighborhood

#### 数据代价 | Data Cost

提供两种匹配代价计算方法：

Two matching cost computation methods are provided:

**SAD（Sum of Absolute Differences）**：

$$C_{\text{SAD}}(p, d) = \frac{1}{|W|} \sum_{q \in W(p)} |I_L(q) - I_R(q - d)|$$

在窗口 $W$ 内计算左右图像素的绝对差之和（通过 `cv2.boxFilter` 实现），简单快速但对光照变化敏感。

Computes the sum of absolute differences between left and right image pixels within window $W$ (implemented via `cv2.boxFilter`). Simple and fast but sensitive to illumination changes.

**Census Transform**（默认 | default）：

将窗口内每个像素与中心像素比较，生成二进制描述符（最多 64 位），左右 Census 码的 **Hamming 距离**作为匹配代价。Census Transform 对光照变化具有鲁棒性，因为它只编码像素间的相对大小关系而非绝对灰度值。

Each pixel within the window is compared with the center pixel to generate a binary descriptor (up to 64 bits). The **Hamming distance** between left and right Census codes serves as the matching cost. Census Transform is robust to illumination changes because it encodes only relative ordering between pixels rather than absolute intensity values.

#### 平滑代价 | Smoothness Cost

采用截断二次距离变换：

A truncated quadratic distance transform is used:

$$V(d_p - d_q) = \min\bigl(\lambda \cdot T^2,\ \lambda \cdot (d_p - d_q)^2\bigr)$$

其中 $T$ 为截断阈值（默认 15.0）。截断机制允许相邻像素的视差在深度不连续处（如物体边缘）发生突变，避免过度平滑。

where $T$ is the truncation threshold (default 15.0). The truncation mechanism allows disparity jumps at depth discontinuities (e.g., object boundaries), preventing over-smoothing.

### 2.2 算法细节 | Algorithm Details

#### 分层 BP（Hierarchical BP）

直接在原始分辨率上运行 BP 收敛慢且效率低。分层 BP 通过构建**高斯金字塔**实现粗到细的多尺度推理：

Running BP directly at full resolution converges slowly and is inefficient. Hierarchical BP achieves coarse-to-fine multi-scale inference through **Gaussian pyramids**:

1. **构建金字塔 | Build pyramid**：对左右灰度图分别构建多层高斯金字塔（默认 4 层），每层分辨率减半（2×2 均值下采样）
   Multi-level Gaussian pyramids are built for both left and right grayscale images (default 4 levels), with resolution halved at each level (2×2 mean downsampling).
2. **粗层推理 | Coarse-level inference**：在最粗层（分辨率最低、视差标签数也按比例缩减为 $D_{\max} / 2^{\text{level}}$）上运行 BP，少量迭代即可覆盖较大范围的空间信息
   BP is run at the coarsest level (lowest resolution, with disparity labels proportionally reduced to $D_{\max} / 2^{\text{level}}$), where a few iterations can cover a large spatial range.
3. **消息上采样 | Message upsampling**：将粗层消息通过最近邻 2× 上采样传递到细层，并在标签维度上进行线性插值重采样，以适应细层的视差标签数量
   Coarse-level messages are upsampled to the finer level via 2× nearest-neighbor upsampling, with linear interpolation resampling along the label dimension to match the finer level's disparity label count.
4. **细层精炼 | Fine-level refinement**：在细层上以上采样的消息为初始值继续 BP 迭代，逐步精炼视差估计
   BP iterations continue at the finer level using the upsampled messages as initialization, progressively refining the disparity estimate.
5. **重复 | Repeat**：直到原始分辨率层完成 | Until the original resolution level is completed.

#### 消息传递 | Message Passing

每层内的 BP 迭代与图像降噪类似：

BP iterations within each level are similar to those in image denoising:

- **4 方向 Min-Sum | 4-directional Min-Sum**：上→下、下→上、左→右、右→左 | up→down, down→up, left→right, right→left
- **截断二次距离变换 | Truncated quadratic distance transform**：采用截断半径优化的 $O(D \times N)$ 扫描算法 | Using a truncation-radius-optimized $O(D \times N)$ sweep algorithm
- **归一化 | Normalization**：消息减去均值（而非最小值），数值更稳定 | Messages normalized by subtracting the mean (instead of minimum), more numerically stable
- **阻尼 | Damping**：$\alpha = 0.5$，抑制振荡 | $\alpha = 0.5$, suppressing oscillations

#### 预处理与后处理 | Pre-processing and Post-processing

- **预处理 | Pre-processing**：对左右灰度图进行轻度高斯模糊（$3 \times 3$，$\sigma = 0.8$），减少噪声对匹配代价的干扰
  Light Gaussian blur ($3 \times 3$, $\sigma = 0.8$) is applied to both grayscale images to reduce noise interference on matching costs.
- **后处理 | Post-processing**：
  - **中值滤波 | Median filter**（$5 \times 5$）：去除视差图中的孤立噪点 | Removes isolated outliers in the disparity map
  - **双边滤波 | Bilateral filter**（$d=7$，$\sigma_{\text{color}}=8$，$\sigma_{\text{space}}=8$）：在保持深度不连续边缘的同时平滑视差 | Smooths disparity while preserving depth discontinuity edges
- **可视化 | Visualization**：使用 JET 色彩映射将视差值着色为深度图 | JET colormap is applied to visualize disparity values as a depth map

#### GPU 加速 | GPU Acceleration

通过 CuPy 库实现 GPU 加速，API 与 NumPy 完全兼容。系统自动检测 GPU 可用性和显存，决定是否启用 GPU 计算。

GPU acceleration is implemented via the CuPy library, whose API is fully compatible with NumPy. The system automatically detects GPU availability and memory to decide whether to enable GPU computation.

### 2.3 效果展示 | Performance Demonstration

在内置合成数据集上，使用默认参数（$D_{\max}=64$，$\lambda=0.1$，$T=15$，4 层金字塔，每层 5 次迭代，Census 代价）的评估结果如下：

On the built-in synthetic datasets, using default parameters ($D_{\max}=64$, $\lambda=0.1$, $T=15$, 4-level pyramid, 5 iterations per level, Census cost), the evaluation results are as follows:

| 场景 Scene | 描述 Description | 评估指标 Evaluation Focus |
|------------|------------------|--------------------------|
| **blocks** | 3 个不同深度的矩形色块（视差 16/32/48）/ 3 rectangular blocks at different depths (disparity 16/32/48) | 深度不连续边界测试 / Depth discontinuity boundary test |
| **corridor** | 走廊透视场景，视差线性渐变 0→60 / Corridor perspective, linear disparity gradient 0→60 | 平滑表面测试 / Smooth surface test |
| **spheres** | 2–3 个球体，视差呈圆形分布 / 2–3 spheres with circular disparity distribution | 曲面深度测试 / Curved surface depth test |

评估采用两个指标：| Two evaluation metrics are used:

- **MAE（Mean Absolute Error）**：与真值视差的平均绝对误差（像素）| Mean absolute error against ground truth disparity (pixels)
- **错误率（Error Rate）**：$|d_{\text{est}} - d_{\text{gt}}| > 2$ 像素的比例 | Percentage of pixels where $|d_{\text{est}} - d_{\text{gt}}| > 2$ pixels

分层 BP 在 blocks 等分段常数场景上表现较好（大片区域的视差可以通过消息传递正确恢复），但在 spheres 的曲面区域和所有场景的深度不连续边界处存在一定误差。

Hierarchical BP performs well on piecewise-constant scenes like blocks (disparity in large regions can be correctly recovered through message passing), but shows certain errors in curved regions of spheres and at depth discontinuity boundaries in all scenes.

### 2.4 BP 的作用 | Role of BP

在立体匹配中，BP 的核心作用是：

In stereo matching, the key roles of BP are:

- **全局视差优化 | Global disparity optimization**：局部匹配代价受噪声、弱纹理、遮挡等因素干扰可能不可靠，BP 通过消息传递在视差空间上进行全局能量最小化，综合周围像素信息修正局部匹配错误。
  Local matching costs may be unreliable due to noise, weak textures, and occlusion. BP performs global energy minimization in disparity space through message passing, integrating surrounding pixel information to correct local matching errors.
- **视差一致性约束 | Disparity consistency enforcement**：平滑项强制相邻像素的视差保持一致，有效抑制弱纹理区域（如白墙、天空）的视差噪声。
  The smoothness term enforces disparity consistency between neighboring pixels, effectively suppressing disparity noise in weakly textured regions (e.g., white walls, sky).
- **多尺度推理 | Multi-scale inference**：分层结构使 BP 能够兼顾全局结构（粗层）和局部细节（细层），大幅提升收敛速度和结果质量。
  The hierarchical structure enables BP to balance global structure (coarse level) and local details (fine level), significantly improving convergence speed and result quality.
- **遮挡与弱纹理处理 | Occlusion and weak texture handling**：虽然没有显式的遮挡推理，但全局优化机制使 BP 能够在一定程度上利用周围可靠区域的信息"填补"遮挡和弱纹理区域的视差。
  Although there is no explicit occlusion reasoning, the global optimization mechanism allows BP to partially "fill in" disparity for occluded and weakly textured regions using information from surrounding reliable areas.

### 2.5 不足与瓶颈 | Limitations and Bottlenecks

1. **计算量大 | High computational cost**：总复杂度为 $O(H \times W \times D \times \text{iters} \times \text{levels})$。即使有分层加速，处理 400×400 图像仍需数秒到数十秒。
   Total complexity is $O(H \times W \times D \times \text{iters} \times \text{levels})$. Even with hierarchical acceleration, processing 400×400 images takes seconds to tens of seconds.
2. **分辨率受限 | Resolution limited**：Web 应用上传图像最大 640px，实际匹配时受计算量限制通常不超过 400px。高分辨率（如 1080p、4K）图像的实时处理无法实现。
   The web application limits uploads to 640px, and practical matching is typically limited to 400px. Real-time processing of high-resolution (e.g., 1080p, 4K) images is infeasible.
3. **无显式遮挡推理 | No explicit occlusion reasoning**：BP 的平滑项假设所有相邻像素都应有相似视差，无法正确处理遮挡区域（左图可见但右图不可见的区域）。
   BP's smoothness term assumes all neighboring pixels should have similar disparity, unable to correctly handle occluded regions (visible in the left image but not in the right).
4. **仅支持整数视差 | Integer disparity only**：视差标签为整数，无法实现亚像素精度。
   Disparity labels are integers, unable to achieve sub-pixel precision.
5. **对大面积无纹理区域鲁棒性有限 | Limited robustness in large textureless regions**：尽管平滑项可以传播邻近视差信息，但大面积无纹理区域（如白墙）的匹配代价几乎无区分度，BP 可能传播错误视差。
   Although the smoothness term can propagate nearby disparity information, matching costs in large textureless regions (e.g., white walls) are nearly indiscriminative, and BP may propagate incorrect disparity.

### 2.6 与前沿方法对比 | Comparison with State-of-the-Art

| 方法 Method | 年份 Year | 类型 Type | Middlebury bad2.0 | 速度 Speed |
|-------------|-----------|-----------|-------------------|------------|
| **Hierarchical BP (ours)** | — | 分层 MRF-BP / Hierarchical MRF-BP | 中等 Medium | ~sec / 400px |
| SGM | 2008 | 半全局动态规划 / Semi-global dynamic programming | 较低 Lower | 实时可达 Real-time capable |
| MC-CNN | 2015 | 学习型匹配代价 / Learned matching cost + SGM | 较低 Lower | ~1s |
| GC-Net | 2017 | 端到端 3D CNN / End-to-end 3D CNN | 低 Low | ~1s (GPU) |
| RAFT-Stereo | 2021 | 光流式迭代优化 / Optical-flow-style iterative refinement | 很低 Very low | ~200ms (GPU) |
| CREStereo | 2022 | 级联循环优化 / Cascaded recurrent refinement | 很低 Very low | ~200ms (GPU) |
| Unimatch | 2023 | 统一匹配框架 / Unified matching framework | 极低 Extremely low | ~300ms (GPU) |

**分析 | Analysis**：

- 在 Middlebury 等主流基准排行榜上，**传统 BP 方法已不在前列**。SGM 以更低的计算复杂度（多方向动态规划 vs. 多次迭代消息传递）取得了更好的效果，成为工业界（如自动驾驶）的标准选择。
  On mainstream benchmarks like Middlebury, **traditional BP methods are no longer at the forefront**. SGM achieves better results with lower computational complexity (multi-directional dynamic programming vs. multi-iteration message passing) and has become the industry standard (e.g., autonomous driving).
- 深度学习方法（RAFT-Stereo、CREStereo）通过端到端训练，将特征提取、匹配代价计算和全局优化统一在可微框架中，在精度和速度上全面超越传统方法。
  Deep learning methods (RAFT-Stereo, CREStereo) unify feature extraction, matching cost computation, and global optimization within a differentiable framework through end-to-end training, surpassing traditional methods in both accuracy and speed.
- 但 BP 立体匹配的**思想影响深远**：许多深度学习方法的 cost aggregation 模块实际上可以解释为神经化的消息传递过程。
  However, the **intellectual influence of BP stereo matching is profound**: the cost aggregation modules in many deep learning methods can be interpreted as neuralized message-passing processes.

---

## 模块三：基于 BP 的蛋白质侧链构象预测 | Module 3: BP-Based Protein Side-Chain Conformation Prediction

> **技术总结 | Technical Summary**
>
> **算法**：SCWRL4（Dead-End Elimination + 图分解）/ TRBP（Tree-Reweighted BP）| **Algorithm**: SCWRL4 (Dead-End Elimination + graph decomposition) / TRBP (Tree-Reweighted BP)
> **图模型**：残基交互图（空间距离 < 阈值的残基对连边），变量 = rotamer 选择 | **Graphical Model**: Residue interaction graph (edges between residue pairs within distance threshold), variables = rotamer choice
> **一元势**：残基自身能量（主链-侧链相互作用 + rotamer 先验）| **Unary**: Residue self-energy (backbone-sidechain interaction + rotamer prior)
> **二元势**：残基对相互作用（范德华力 + 静电）| **Pairwise**: Residue pair interaction (van der Waals + electrostatics)
> **关键技术**：DEE 剪枝 + 图连通分量分解 + ROSETTA 力场 + TRBP 生成树凸组合 | **Key Techniques**: DEE pruning + graph connected component decomposition + ROSETTA force field + TRBP spanning tree convex combination
> **推理目标**：MAP 估计（全局能量最小的 rotamer 组合）| **Inference Goal**: MAP estimation (globally minimal energy rotamer assignment)

### 案例：SCWRL 系列与 TRBP 方法 | Case Study: SCWRL Series and TRBP Methods

本模块聚焦蛋白质结构预测中的一个经典子问题——**侧链构象预测（Side-Chain Prediction）**，分析 SCWRL4 和 TRBP（Tree-Reweighted BP）方法中 BP 的应用。

This module focuses on a classic sub-problem in protein structure prediction — **side-chain conformation prediction**, analyzing the application of BP in the SCWRL4 and TRBP (Tree-Reweighted BP) methods.

### 3.1 问题描述 | Problem Description

蛋白质由氨基酸残基组成，每个残基包含**主链**（backbone，即 N-Cα-C 骨架）和**侧链**（side chain）。当主链构象已知（如通过实验测定或结构预测获得）时，需要预测每个残基侧链的三维构象。

Proteins are composed of amino acid residues, each containing a **backbone** (the N-Cα-C skeleton) and a **side chain**. When the backbone conformation is known (e.g., from experimental determination or structure prediction), the task is to predict the 3D conformation of each residue's side chain.

侧链构象可以用一组**二面角**（dihedral angles）$\chi_1, \chi_2, \dots$ 来描述。由于侧链二面角倾向于聚集在特定角度附近（受化学键旋转势垒约束），人们将这些优势构象离散化为**旋转异构体（rotamer）**库。侧链预测问题因此归结为：

Side-chain conformations can be described by a set of **dihedral angles** $\chi_1, \chi_2, \dots$. Since side-chain dihedral angles tend to cluster around specific angles (constrained by rotational energy barriers of chemical bonds), these preferred conformations are discretized into a **rotamer library**. The side-chain prediction problem thus reduces to:

> 给定主链坐标和序列，为每个残基从 rotamer 库中选择最佳构象，使整体能量最低。
> Given backbone coordinates and sequence, select the optimal conformation from the rotamer library for each residue to minimize the overall energy.

这是一个典型的**组合优化**问题。对于包含 $N$ 个残基、平均每个残基有 $K$ 个 rotamer 候选的蛋白质，搜索空间大小为 $K^N$，暴力枚举不可行。

This is a classic **combinatorial optimization** problem. For a protein with $N$ residues and an average of $K$ rotamer candidates per residue, the search space is $K^N$, making brute-force enumeration infeasible.

### 3.2 图模型建模 | Graphical Model Formulation

将侧链预测建模为 MRF / 因子图：

Side-chain prediction is modeled as an MRF / factor graph:

- **变量节点 | Variable nodes**：每个残基 $i$ 对应一个离散变量 $r_i$，取值空间为该残基的 rotamer 库 $\{r_i^1, r_i^2, \dots, r_i^{K_i}\}$
  Each residue $i$ corresponds to a discrete variable $r_i$, with value space being the residue's rotamer library $\{r_i^1, r_i^2, \dots, r_i^{K_i}\}$.
- **一元势函数 | Unary potential**：残基 $i$ 选择 rotamer $r_i^k$ 时的自身能量（主链-侧链相互作用 + rotamer 先验概率）
  Self-energy of residue $i$ when choosing rotamer $r_i^k$ (backbone-sidechain interaction + rotamer prior probability).
- **二元势函数 | Pairwise potential**：残基 $i$ 和 $j$ 分别选择 $r_i^k$ 和 $r_j^l$ 时的对相互作用能量（范德华力、静电相互作用等）
  Pairwise interaction energy when residues $i$ and $j$ choose $r_i^k$ and $r_j^l$ respectively (van der Waals forces, electrostatic interactions, etc.).
- **图结构 | Graph structure**：**残基交互图**——空间距离小于阈值（通常 ~5–10 Å）的残基对之间连边
  **Residue interaction graph** — edges connect residue pairs with spatial distance below a threshold (typically ~5–10 Å).

整体能量函数为：| The overall energy function is:

$$E(\mathbf{r}) = \sum_i E_{\text{self}}(r_i) + \sum_{(i,j) \in \mathcal{E}} E_{\text{pair}}(r_i, r_j)$$

目标是求 $\arg\min_{\mathbf{r}} E(\mathbf{r})$，即全局能量最小的 rotamer 组合。

The goal is to find $\arg\min_{\mathbf{r}} E(\mathbf{r})$, i.e., the rotamer combination with the globally minimum energy.

### 3.3 算法细节 | Algorithm Details

#### SCWRL4：Dead-End Elimination + 图分解 | SCWRL4: Dead-End Elimination + Graph Decomposition

SCWRL4 (2009) 是侧链预测领域的经典基准方法，采用多步策略：

SCWRL4 (2009) is a classic benchmark in side-chain prediction, employing a multi-step strategy:

1. **Dead-End Elimination (DEE)**：通过能量下界分析，剪除不可能出现在最优解中的 rotamer。如果某个 rotamer $r_i^k$ 在任何情况下的能量都高于同一残基的另一个 rotamer $r_i^l$，则 $r_i^k$ 可以安全删除。
   Through energy lower-bound analysis, rotamers that cannot appear in the optimal solution are pruned. If a rotamer $r_i^k$ always has higher energy than another rotamer $r_i^l$ of the same residue under any configuration, then $r_i^k$ can be safely eliminated.
2. **图分解 | Graph decomposition**：将剪枝后的残基交互图分解为若干连通分量，对小分量进行精确枚举，对大分量使用近似算法。
   The pruned residue interaction graph is decomposed into connected components; small components are solved exactly by enumeration, while large components use approximate algorithms.
3. **近似求解 | Approximate solving**：对无法精确求解的分量，使用基于树分解和消息传递的近似推理。
   For components that cannot be solved exactly, tree decomposition and message-passing-based approximate inference is used.

SCWRL4 在标准测试集上的准确率：| SCWRL4 accuracy on standard test sets:

- $\chi_1$ 准确率 accuracy：约 **82.6%**
- $\chi_{1+2}$ 准确率 accuracy：约 **73.7%**
- 处理速度 Speed：大多数蛋白质在数秒内完成 | Most proteins processed within seconds

#### TRBP：Tree-Reweighted BP

Yanover 等人 (2006) 将 **Tree-Reweighted BP (TRBP)** 应用于 ROSETTA 能量函数下的侧链预测：

Yanover et al. (2006) applied **Tree-Reweighted BP (TRBP)** to side-chain prediction under the ROSETTA energy function:

1. **树分解重加权 | Tree-reweighted decomposition**：将残基交互图分解为多棵生成树的凸组合，每棵树上 BP 精确求解。对所有树的结果进行重加权平均，得到原图上的近似边缘分布。
   The residue interaction graph is decomposed into a convex combination of spanning trees, with BP solved exactly on each tree. Results from all trees are reweighted and averaged to obtain approximate marginals on the original graph.
2. **能量函数 | Energy function**：采用 ROSETTA 分子力场，包含范德华相互作用、氢键、溶剂化效应等。
   Uses the ROSETTA molecular force field, including van der Waals interactions, hydrogen bonds, solvation effects, etc.
3. **收敛保证 | Convergence guarantee**：相比普通 Loopy BP，TRBP 有更好的理论性质——其 Bethe 自由能是真实自由能的上界，优化过程保证单调递减。
   Compared to standard Loopy BP, TRBP has better theoretical properties — its Bethe free energy is an upper bound on the true free energy, and the optimization process is guaranteed to be monotonically decreasing.

TRBP 方法的表现：| TRBP performance:

- 约 **85%** 的蛋白质能找到全局能量最优解 | ~**85%** of proteins find the global energy optimum
- 在剩余蛋白质上也能给出接近最优的解 | Near-optimal solutions for the remaining proteins
- 与 SCWRL3 相比，准确率提升显著 | Significant accuracy improvement over SCWRL3

### 3.4 BP 的作用 | Role of BP

在侧链构象预测中，BP 的核心作用是：

In side-chain conformation prediction, the key roles of BP are:

- **高效处理组合爆炸 | Efficiently handling combinatorial explosion**：$N$ 个残基各有 $K$ 个 rotamer，搜索空间 $K^N$ 是指数级的。BP 通过局部消息传递进行近似全局优化，时间复杂度仅为 $O(N \cdot K^2 \cdot \text{iters})$，远优于精确枚举。
  With $N$ residues each having $K$ rotamers, the search space $K^N$ is exponential. BP performs approximate global optimization through local message passing with time complexity of only $O(N \cdot K^2 \cdot \text{iters})$, far better than exact enumeration.
- **捕获残基间协同效应 | Capturing inter-residue cooperative effects**：残基 A 的侧链构象会影响其空间邻居 B 的可选构象（因为侧链间可能发生空间冲突）。BP 的消息传递机制天然编码了这种协同效应。
  The side-chain conformation of residue A affects the available conformations of its spatial neighbor B (due to potential steric clashes). BP's message-passing mechanism naturally encodes such cooperative effects.
- **稀疏图上的高效推理 | Efficient inference on sparse graphs**：残基交互图通常是稀疏的（每个残基仅与空间邻近的少数残基相互作用），BP 在稀疏图上效率极高。
  Residue interaction graphs are typically sparse (each residue interacts with only a few spatially proximate residues), and BP is highly efficient on sparse graphs.
- **提供概率信息 | Providing probabilistic information**：除了 MAP 估计外，BP 还可以给出每个残基各 rotamer 的概率（置信度），这对识别构象不确定的残基非常有价值。
  Beyond MAP estimation, BP can also provide the probability (belief) of each rotamer for every residue, which is valuable for identifying residues with conformational uncertainty.

### 3.5 不足与瓶颈 | Limitations and Bottlenecks

1. **依赖离散 rotamer 库 | Dependence on discrete rotamer libraries**：侧链二面角在自然界中是连续值，离散化为有限的 rotamer 集合不可避免地引入量化误差。虽然可以通过增加 rotamer 数量来提高分辨率，但这会显著增加计算复杂度。
   Side-chain dihedral angles are continuous in nature; discretization into a finite rotamer set inevitably introduces quantization error. Increasing the number of rotamers improves resolution but significantly increases computational cost.
2. **能量函数精度受限 | Limited energy function accuracy**：经验分子力场（如 ROSETTA 的 Lennard-Jones 势）相比量子力学计算仍有较大近似误差。能量函数的不准确会直接导致最优 rotamer 选择的偏差。
   Empirical molecular force fields (e.g., ROSETTA's Lennard-Jones potential) still have significant approximation errors compared to quantum mechanical calculations. Inaccuracies in the energy function directly lead to biased optimal rotamer selection.
3. **含环图上的近似性 | Approximation on loopy graphs**：残基交互图通常包含环（多个残基形成空间邻近的"团"），Loopy BP 在此类图上仅提供近似解，且精度难以控制。TRBP 虽提供更好的理论保证，但仍为近似推理。
   Residue interaction graphs typically contain loops (multiple residues forming spatially proximate "clusters"). Loopy BP provides only approximate solutions on such graphs with hard-to-control accuracy. Although TRBP offers better theoretical guarantees, it is still approximate inference.
4. **忽略主链柔性 | Ignoring backbone flexibility**：传统方法假设主链坐标固定不变，但实际上侧链构象的改变也会引起主链的微调。忽略这种耦合效应会降低预测精度。
   Traditional methods assume fixed backbone coordinates, but in reality, changes in side-chain conformations also induce backbone adjustments. Ignoring this coupling effect reduces prediction accuracy.
5. **不擅长处理无序区域 | Poor handling of disordered regions**：蛋白质中存在固有无序区域（intrinsically disordered regions），这些区域不存在单一稳定构象，而 MAP 推理天然假设存在"最优"构象。
   Proteins contain intrinsically disordered regions (IDRs) where no single stable conformation exists, yet MAP inference inherently assumes the existence of an "optimal" conformation.

### 3.6 与前沿方法对比 | Comparison with State-of-the-Art

| 方法 Method | 年份 Year | 类型 Type | 核心技术 Core Technology | χ₁ 准确率 Accuracy | 速度 Speed |
|-------------|-----------|-----------|--------------------------|---------------------|------------|
| **SCWRL4** | 2009 | DEE + 图分解 Graph decomposition | 组合优化 Combinatorial optimization | ~82.6% | 秒级 Seconds |
| **TRBP** | 2006 | Tree-Reweighted BP | 消息传递 Message passing | — (85% find optimum) | 秒级 Seconds |
| OSCAR-star | 2011 | 迭代优化 Iterative optimization | 方向依赖势 Orientation-dependent potential | ~86% | 秒级 Seconds |
| DLPacker | 2022 | 3D U-Net | 深度学习 Deep learning | ~87% | 亚秒级 Sub-second |
| AttnPacker | 2023 | Attention 网络 Network | 深度学习 Deep learning | ~89% | 亚秒级 Sub-second |
| AlphaFold2 | 2021 | Evoformer + Structure Module | 端到端深度学习 End-to-end DL | >90% (side-chain) | 分钟级 Minutes |
| AlphaFold3 | 2024 | Diffusion + Pairformer | 端到端深度学习 End-to-end DL | 更高 Higher | 分钟级 Minutes |

**分析 | Analysis**：

- **DLPacker (2022)**：使用 3D U-Net 在密度图空间中直接预测侧链原子坐标，RMSD 比 SCWRL4 平均小约 20%，且对 rotamer 库没有依赖（直接预测连续坐标）。
  Uses 3D U-Net to directly predict side-chain atom coordinates in density map space. RMSD is ~20% lower than SCWRL4 on average, with no dependence on rotamer libraries (directly predicts continuous coordinates).
- **AttnPacker (2023)**：利用 Attention 机制建模残基间相互作用，直接预测侧链原子坐标，速度比传统方法快 100 倍以上，精度也更高。
  Uses Attention mechanism to model inter-residue interactions, directly predicting side-chain atom coordinates. Over 100x faster than traditional methods with higher accuracy.
- **AlphaFold2/3**：端到端预测蛋白质全原子结构（包括主链和侧链），已从根本上改变了蛋白质结构预测的范式。侧链预测不再是独立任务，而是整体结构预测的一部分。
  End-to-end prediction of full-atom protein structures (including backbone and side chains), fundamentally transforming the paradigm of protein structure prediction. Side-chain prediction is no longer a standalone task but part of holistic structure prediction.
- 传统 BP 方法在精度上落后于深度学习方法 5–10 个百分点，在速度上也不占优势。但 BP 方法的**物理可解释性**（能量函数有明确物理含义）和**不依赖训练数据**的特点仍有理论价值。
  Traditional BP methods lag behind deep learning methods by 5–10 percentage points in accuracy and have no speed advantage. However, BP methods retain theoretical value due to their **physical interpretability** (energy functions have clear physical meaning) and **independence from training data**.

---

## 总结与展望 | Summary and Outlook

### BP 的历史地位 | Historical Significance of BP

置信传播作为概率图模型上的经典推理算法，在过去四十年中为计算机视觉、信号处理、通信编码和计算生物学等领域提供了统一的理论框架。它的核心思想——**通过局部消息传递实现全局推理**——至今仍是理解和设计智能算法的重要范式。

As a classic inference algorithm on probabilistic graphical models, belief propagation has provided a unified theoretical framework for computer vision, signal processing, channel coding, and computational biology over the past four decades. Its core idea — **achieving global inference through local message passing** — remains an important paradigm for understanding and designing intelligent algorithms.

### 三个应用中的共同模式 | Common Patterns Across Three Applications

从本报告分析的三个应用中，可以抽象出 BP 解决实际问题的共同范式：

From the three applications analyzed in this report, a common paradigm of BP for solving real-world problems can be abstracted:

| 步骤 Step | 图像降噪 Image Denoising | 立体匹配 Stereo Matching | 蛋白质侧链预测 Protein Side-Chain |
|-----------|--------------------------|--------------------------|-----------------------------------|
| **局部交互建模 Local interaction modeling** | 相邻像素灰度值应相似 Neighboring pixel intensities should be similar | 相邻像素视差应一致 Neighboring pixel disparities should be consistent | 空间邻近残基侧链不应冲突 Spatially proximate residue side chains should not clash |
| **MRF 构建 MRF construction** | 4-连接网格 4-connected grid | 4-连接网格 4-connected grid | 残基交互图 Residue interaction graph |
| **变量空间 Variable space** | 量化灰度标签 Quantized intensity labels (L=64) | 整数视差标签 Integer disparity labels (D≤64) | rotamer 库 Rotamer library (K~10–100) |
| **一元势 Unary potential** | 高斯观测似然 Gaussian observation likelihood | SAD / Census 匹配代价 matching cost | 自身能量 + rotamer 先验 Self-energy + rotamer prior |
| **二元势 Pairwise potential** | 截断二次/线性模型 Truncated quadratic/linear | 截断二次距离变换 Truncated quadratic DT | 范德华力 + 静电作用 vdW + electrostatics |
| **BP 变体 BP variant** | Min-Sum Loopy BP | Hierarchical Min-Sum BP | TRBP / Loopy BP |
| **输出 Output** | MAP 灰度值 MAP intensity | MAP 视差图 MAP disparity map | MAP rotamer 组合 MAP rotamer assignment |

### 深度学习时代 BP 的价值 | The Value of BP in the Deep Learning Era

尽管在上述三个应用中，深度学习方法已在精度和速度上全面超越传统 BP，但 BP 仍具有独特的价值：

Although deep learning methods have comprehensively surpassed traditional BP in both accuracy and speed across all three applications above, BP retains unique value:

1. **可解释性 | Interpretability**：BP 的每一步操作都有明确的概率含义（消息 = 边缘似然的近似），便于理解和调试。而深度学习模型通常是黑箱。
   Every step of BP has a clear probabilistic interpretation (messages ≈ approximate marginal likelihoods), facilitating understanding and debugging. Deep learning models are typically black boxes.
2. **理论保证 | Theoretical guarantees**：在树结构图上 BP 精确收敛；TRBP 提供自由能上界保证。深度学习方法通常缺乏类似的理论保证。
   BP converges exactly on tree-structured graphs; TRBP provides free energy upper bound guarantees. Deep learning methods typically lack such theoretical guarantees.
3. **无需训练数据 | No training data required**：BP 基于手工设计的能量函数，不需要标注数据进行训练。在数据稀缺的领域（如特殊工业场景、罕见蛋白质）这一优势明显。
   BP is based on hand-crafted energy functions and requires no labeled data for training. This advantage is significant in data-scarce domains (e.g., specialized industrial scenarios, rare proteins).
4. **与神经网络的结合（Neural BP）| Integration with neural networks (Neural BP)**：近年来出现了将 BP 的消息传递结构与神经网络参数化相结合的方法：
   Recent years have seen methods combining BP's message-passing structure with neural network parameterization:
   - **Neural Belief Propagation**：用神经网络参数化 BP 的势函数和消息更新规则，兼具 BP 的结构归纳偏置和神经网络的表达能力。
     Neural networks parameterize BP's potential functions and message update rules, combining BP's structural inductive bias with neural network expressiveness.
   - **Graph Neural Networks (GNN)**：GNN 中的消息传递机制（message passing）本质上与 BP 同构，可以看作 BP 在连续空间上的推广。
     The message-passing mechanism in GNNs is essentially isomorphic to BP and can be viewed as BP's generalization to continuous spaces.
   - **Unrolled BP**：将固定次数的 BP 迭代展开为神经网络层，端到端训练势函数参数。
     A fixed number of BP iterations are unrolled into neural network layers, with potential function parameters trained end-to-end.

BP 从"直接解决问题的工具"逐渐转变为"设计深度学习架构的思想源泉"。理解 BP 的原理，对于理解和改进现代图神经网络、注意力机制、扩散模型等仍有重要意义。

BP has gradually evolved from "a direct problem-solving tool" to "an intellectual source for designing deep learning architectures." Understanding BP's principles remains significant for understanding and improving modern graph neural networks, attention mechanisms, diffusion models, and beyond.

---

## 参考文献 | References

1. Pearl, J. (1982). Reverend Bayes on inference engines: A distributed hierarchical approach. *AAAI*.
2. Felzenszwalb, P. F., & Huttenlocher, D. P. (2006). Efficient belief propagation for early vision. *IJCV*, 70(1), 41–54.
3. Wainwright, M. J., Jaakkola, T. S., & Willsky, A. S. (2005). MAP estimation via agreement on trees: message-passing and linear programming. *IEEE Trans. IT*, 51(11), 3697–3717.
4. Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). Image denoising by sparse 3-D transform-domain collaborative filtering (BM3D). *IEEE Trans. IP*, 16(8), 2080–2095.
5. Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising (DnCNN). *IEEE Trans. IP*, 26(7), 3142–3155.
6. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). SwinIR: Image restoration using Swin Transformer. *ICCV Workshops*.
7. Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., & Yang, M. H. (2022). Restormer: Efficient Transformer for high-resolution image restoration. *CVPR*.
8. Hirschmüller, H. (2008). Stereo processing by semiglobal matching and mutual information (SGM). *IEEE Trans. PAMI*, 30(2), 328–341.
9. Lipson, L., Teed, Z., & Deng, J. (2021). RAFT-Stereo: Multilevel recurrent field transforms for stereo matching. *3DV*.
10. Li, J., Wang, P., Xiong, P., Cai, T., Yan, Z., Yang, L., ... & Liu, Z. (2022). Practical stereo matching via cascaded recurrent network with adaptive correlation (CREStereo). *CVPR*.
11. Krivov, G. G., Shapovalov, M. V., & Dunbrack, R. L. Jr. (2009). Improved prediction of protein side-chain conformations with SCWRL4. *Proteins*, 77(4), 778–795.
12. Yanover, C., Schueler-Furman, O., & Weiss, Y. (2008). Minimizing and learning energy functions for side-chain prediction. *Journal of Computational Biology*, 15(7), 899–911.
13. Misiura, M., Shroff, R., & Thyer, R. (2022). DLPacker: Deep learning for prediction of amino acid side chain conformations in proteins. *Proteins*, 90(6), 1278–1290.
14. Zhang, Z., Xu, M., Jamasb, A. R., Chenthamarakshan, V., Lozano, A., Das, P., & Tang, J. (2023). Protein representation learning by geometric structure pretraining (AttnPacker). *arXiv*.
15. Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.
16. Abramson, J., Adler, J., Dunger, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630, 493–500.
