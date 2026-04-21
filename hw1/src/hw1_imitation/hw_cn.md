# Berkeley CS 185/285 深度强化学习、决策与控制

## 作业 1：模仿学习

**截止时间：** 2 月 11 日晚上 11:59

## 1. 引言

在这次作业中，你将为 Push-T 环境训练 action chunking policy。你首先会训练一个简单的 MSE（均方误差）策略，它可以通过一次前向传播预测一段动作序列。随后，你将训练一个表达能力更强的 flow matching 策略，它与 diffusion policy（Chi et al., 2025）类似。

**图 1：** Push-T 环境。观测是一个 5 维状态，描述了 T 形物块和 agent 的位置。动作是一个 2 维向量，表示 agent 的目标位置。任务目标是将 T 形物块推入目标区域。

## 2. 使用 MSE 损失的 Action Chunking

Action chunking 的作用是通过一次预测一小段未来动作来降低决策频率。在时间步 $t$，策略 $\pi_\theta(A_t \mid o_t)$ 将当前观测 $o_t$ 映射为一个动作块

$$
A_t = (a_t, a_{t+1}, \ldots, a_{t+K-1})
$$

其中 $K$ 是固定的 chunk 长度。这个动作块会以 open-loop 的方式执行：环境在时间 $t$ 接收 $a_t$，在时间 $t + 1$ 接收 $a_{t+1}$，依此类推，直到执行完 $a_{t+K-1}$。当这个 chunk 执行结束后，策略会再次基于最新观测 $o_{t+K}$ 输出下一个 chunk。

训练 action chunking policy 的最简单方法是使用均方误差（MSE）损失。也就是说，给定由观测和专家动作块组成的数据集 $(o_t^{(j)}, A_t^{(j)})$，我们通过最小化如下目标来拟合 $\pi_\theta$：

$$
L_{\text{MSE}}(\theta)
= \frac{1}{B} \sum_{j=1}^{B}
\left\| A_t^{(j)} - \pi_\theta\!\left(o_t^{(j)}\right) \right\|_2^2
$$

其中 $B$ 是 batch size，$\pi_\theta(o_t^{(j)})$ 表示策略网络的输出。

### 2.1 实现部分

这部分作业要求你实现 MSE policy 和主训练循环。

本次作业的 starter code 位于：

[https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw1](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw1)

我们建议你先认真阅读以下文件：

- `README.md`：介绍项目的基本配置方法。
- `src/hw1_imitation/data.py`：定义数据集类，并负责 Push-T 数据集的下载、解压与加载。
- `src/hw1_imitation/model.py`：定义 `BasePolicy` 类。
- `src/hw1_imitation/evaluation.py`：定义 `evaluate_policy` 函数。你不需要修改这个文件，但你需要理解如何在训练循环中定期调用它。

你需要完成的内容：

- 在 `src/hw1_imitation/model.py` 中补全 TODO，实现 `MSEPolicy` 类。推荐使用一个带 ReLU 激活的简单 MLP。
- 在 `src/hw1_imitation/train.py` 中补全 TODO，实现主训练循环。
- 在训练循环中定期调用 `evaluate_policy`，将评估指标和视频记录到 WandB。

需要提交的内容：

- 一个可运行的训练循环和 MSE policy。
- 成功训练得到的 WandB 日志（包含视频）。MSE policy 应当至少达到 `0.5` 的 reward。
- 注意：为了评分，你必须在训练循环中定期调用 `evaluate_policy`，并且在训练结束时调用 `logger.dump_for_grading()`。
- 一份简短报告，内容包括：
  - 你最优 MSE policy 的训练曲线（训练步数 vs. loss 和 reward）。请自己生成图，而不是直接截图 WandB。
  - 对你的 MLP 结构做简要说明（层数、隐藏层维度、激活函数等）。

提示：

- 一个较小的 MLP 通常在笔记本 CPU 上几分钟内就能训练完成。
- 使用 PyTorch 内置的 Adam 优化器（Kingma, 2014）。
- 可以对 train step 使用 `torch.compile`，这样会显著提速。
- 除了必须记录的评估指标外，你也可以在 WandB 中记录其他有用指标，例如 loss 和训练速度。

## 3. 使用 Flow Matching 的 Action Chunking

MSE policy 是一次性直接预测整个动作块的，这在面对复杂、可能是多模态的动作块分布时会有困难。Flow matching 通过学习一个条件向量场，把噪声逐步变换为真实动作块。它与 diffusion 很相似，但实现更简单，通常表现也更好。

设 $A_t^{(j)}$ 是一个动作块，$A_{t,0}^{(j)} \sim \mathcal{N}(0, I)$ 是与其同形状的噪声。我们首先采样一个 flow-matching 时间步 $\tau^{(j)} \sim U(0, 1)$，并定义插值：

$$
A_{t,\tau}^{(j)} = \tau^{(j)} A_t^{(j)} + \left(1 - \tau^{(j)}\right) A_{t,0}^{(j)}.
$$

然后训练一个网络 $v_\theta$ 来预测将 $A_{t,\tau}^{(j)}$ 推向 $A_t^{(j)}$ 的速度，使用的 flow-matching 损失为：

$$
L_{\text{FM}}(\theta)
= \frac{1}{B} \sum_{j=1}^{B}
\left\|
v_\theta\!\left(o_t^{(j)}, A_{t,\tau}^{(j)}, \tau^{(j)}\right)
- \left(A_t^{(j)} - A_{t,0}^{(j)}\right)
\right\|_2^2.
$$

在推理阶段，我们先采样初始噪声 $A_{t,0} \sim \mathcal{N}(0, I)$，然后对如下 ODE 进行积分：

$$
\frac{dA_{t,\tau}}{d\tau} = v_\theta(o_t, A_{t,\tau}, \tau)
$$

积分区间为 $\tau = 0$ 到 $\tau = 1$。最简单的积分方法是 Euler 积分，其更新规则为：

$$
A_{t,\tau + \frac{1}{n}}
:= A_{t,\tau} + \frac{1}{n} \cdot v_\theta(o_t, A_{t,\tau}, \tau),
$$

这个更新会重复执行 $n$ 次，从 $\tau = 0$ 一直到 $\tau = 1$，得到 $A_{t,1}$。其中 $n$ 是积分步数（也可称为 denoising steps）。最终的 $A_{t,1} = A_t$，也就是要以 open-loop 方式执行的动作块。

### 3.1 实现部分

这一部分你只需要在 `model.py` 中补全 TODO，实现 `FlowMatchingPolicy` 类。推荐使用与 MSE policy 相同的 MLP 结构。

需要提交的内容：

- 一个可运行的训练循环和 flow matching policy。
- 成功训练得到的 WandB 日志（包含视频）。Flow matching policy 应当至少达到 `0.7` 的 reward。
- 一份简短报告，内容包括：
  - 你最优 flow matching policy 的训练曲线（训练步数 vs. loss 和 reward）。请自己生成图，而不是直接截图 WandB。
  - 基于视频，对 flow matching policy 相比 MSE policy 的行为做一个简要的定性描述。

提示：

- 不要忘记把 flow matching 的时间步 $\tau$ 作为神经网络输入的一部分。

## 4. 代码与实验结果提交方式

为了提交代码和实验日志，请创建一个目录，其中包含以下内容：

- 一个名为 `exp` 的目录，存放本次作业中你最好的实验结果。实验目录最初可能会命名为 `seed_42_20260119_161512_my_experiment_name` 这种形式；你需要按下面要求重命名，并且每一部分只保留你最好的那次运行。
- 一个 `src` 目录，其中包含所有 `.py` 文件，文件名和目录结构应与原始作业仓库保持一致。

解压后的提交内容应当具有如下目录结构。请确保 `exp` 目录的结构与下面完全一致，并且复制整个 run 目录，包括所有 `.wandb`、`.json`、`.mp4` 和 `.pkl` 文件。

```text
submit.zip
├── exp
│   ├── mse
│   │   ├── log.csv
│   │   └── wandb
│   │       └── ...
│   └── flow
│       ├── log.csv
│       └── wandb
│           └── ...
├── src
│   └── hw1_imitation
│       ├── model.py
│       ├── train.py
│       ├── data.py
│       └── evaluation.py
├── pyproject.toml
├── uv.lock
└── README.md
```

请在 Gradescope 提交作业。将包含代码和日志文件的 zip 上传到 **HW1 Code**，将报告上传到 **HW1 Report**。

## 参考文献

- Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. *Diffusion policy: Visuomotor policy learning via action diffusion*. *The International Journal of Robotics Research*, 44(10-11):1684-1704, 2025.
- Diederik P. Kingma. *Adam: A method for stochastic optimization*. *arXiv preprint arXiv:1412.6980*, 2014.
