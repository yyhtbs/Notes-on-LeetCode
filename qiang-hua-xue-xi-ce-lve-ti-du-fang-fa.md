---
description: Some Notes of RL in Chinese
---

# 强化学习-策略梯度方法

这是对策略梯度算法的介绍。 

研究界看到了更多有希望的结果。 有了足够的动力，让我们现在来看一下强化学习问题。 强化学习是对学习问题的一个笼统的描述，其**目的**是**最大化长期目标**。 系统描述由一个**个体**组成，该个体在不连续的时间步长通过其行为与**环境**交互并获得回报。 这会将代理转换为新状态。 下图\[1\]描绘了规范的**个体-环境**反馈回路。

![](.gitbook/assets/image%20%285%29.png)

背景和定义 RL背后的大量理论都基于“奖励假说” \[1\]的假设，该假说概括地说，可以通过称为奖励的单个标量来解释代理的所有目标和目的。 这仍然有争议，但是还很难反驳。 更正式地说，奖励假设如下

奖励假说：我们可以将目标和目的所指的所有含义充分地理解为所接收标量信号（称为奖励）的累加总和的期望值的最大化。

作为RL从业者和研究人员，一个人的工作是为给定问题（称为奖励塑造）找到正确的奖励集。

代理人必须通过称为“马尔可夫决策过程”的理论框架进行正式工作，该过程由在每个州要做出的决策（采取什么措施？）组成。 这产生了一系列状态，动作和奖励，称为轨迹。

 $$S_0 ,A_0 ​,R_1 ​,S_1 ​,A_1 ​,R_2 ​,…$$ 

目的是使这套奖励最大化。 更正式地说，我们看一下马尔可夫决策过程\[1，2\]框架

马尔可夫决策过程：（折扣）马尔可夫决策过程（MDP）是一个元组（ $$\mathcal {S}, \mathcal {A},  \mathcal {R}, p, \gamma$$\)这样

$$
p(s',r∣s,a)=Pr[S _{t+1}=s'  ,R_{ t+1}=r∣S_{t}​=s, A_{t}=a]
$$

$$
G_{t} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

where $$S_{t}, S_{t+1} \in \mathcal{S}$$ \(状态空间\), $$A_{t+1} \in \mathcal{A}$$  \(动作空间\), $$R_{t+1},R_{t} \in \mathcal{R}$$ \(回报空间\), $$p$$ 定义了动态过程，$$G_t$$是折现收益.

简而言之，MDP定义了转换为新状态的可能性，在当前状态和执行某项操作的情况下，它会获得一定的回报。这个框架在数学上令人愉悦，因为它是一阶马尔可夫模型。这只是一种不准确的假设，即接下来发生的一切都仅取决于现在而不是过去。该框架的另一个重要部分是折扣因子$$\gamma$$。随着时间的推移对这些奖励的求和对未来的奖励的重视程度不同，这会导致折现收益的概念。正如人们可能期望的那样，更高的$$\gamma$$会导致对未来奖励的更高敏感性。但是，$$\gamma=0$$的极端情况根本不考虑来自未来的回报。

环境 $$p$$ 的动态超出了个体的控制范围。要对此进行内在化，请想象一下在有风的环境中站在田野上，并每秒在四个方向之一上迈出一步。风是如此之大，以至于您很难沿着与北，东，西或南完全一致的方向移动。在下一秒降落到新状态的可能性由风场的动态 $$p$$ 给出。它肯定不在您（个体）的控制之下。

但是，如果您以某种方式了解环境的动态并朝北，东，西或南以外的方向移动，该怎么办。该策略是代理控制的。当代理遵循策略 $$\pi$$ 时，它会生成状态，动作和奖励的序列，称为“轨迹”。

策略：策略定义为给定状态下动作的概率分布:

$$
\pi(A_t = a | S_t = s) \Leftrightarrow \pi: s \rightarrow P(a|s)
$$

## 策略梯度

强化学习代理的目标是在遵循策略 $$\pi$$ 时最大程度地提高“预期”回报。 像任何机器学习设置一样，我们定义了一组参数 $$\theta$$ （类似复杂多项式的系数或神经网络中单位的权重和偏差）来**参数化此策略** $$\pi_\theta$$（为简便起见，也写为 $$\pi$$ ）。 如果我们将给定轨迹 $$\tau$$ 的总奖励表示为 $$r_{\tau}$$ ，我们得出以下定义。

**强化学习的目标**：通过一个参数化的策略，来最大化“预期”期望回报

$$
J(\theta) = \mathbb{E}_\pi\left[ r(\tau) \right]
$$

所有有限的MDP都有至少一个最优策略（可以提供最大的回报），并且在所有最优策略中，至少有一个是固定的和确定性的\[2\]。

像其他任何机器学习问题一样，目标是找到参数 $$\theta ^ \star$$ 最大化 $$J$$ 。 解决这种最大化问题的标准方法是**梯度上升（或下降）**。 在渐变上升中，我们使用以下更新规则来逐步调整参数

$$
\theta_{t+1} = \theta_{t} + \alpha \nabla J (\theta_{t})
$$

这里的挑战是如何找到包含期望值的目标的梯度。计算期望需要对行为进行遍历和积分，这对计算来说是困难的。 我们需要找到简化的方法。 第一步是通过扩展期望函数，重新构造梯度。

$$
\begin{aligned} \nabla \mathbb{E}_\pi \left[ r(\tau) \right] &= \nabla \int \pi(\tau) r(\tau) d\tau \\ &= \int \nabla\pi(\tau) r(\tau) d\tau \\ &= \int \pi(\tau) \nabla \log \pi(\tau) r(\tau) d\tau \\ \nabla \mathbb{E}_\pi \left[ r(\tau) \right] &= \mathbb{E}_\pi \left[ r(\tau) \nabla \log \pi(\tau) \right] \end{aligned}
$$

这里是通过构造的方法产生 $$\log$$ 的，第3个等号之后的式子可以推回到第2个式子去， $$\nabla\log\pi(\tau)=1/\pi(\tau)\times\nabla{\pi(\tau)}$$ 

策略梯度定理\[1\]：预期回报函数的导数是**策略**的梯度（$$\log\pi_ \theta$$**）**与**回报** $$r$$ 的乘积的期望。

$$
\nabla \mathbb{E}_{\pi_\theta} \left[ r(\tau) \right] = \mathbb{E}_{\pi_\theta} \left[ r(\tau) \nabla \log \pi_\theta(\tau) \right]
$$

现在，让我们扩展策略 $$\pi_ {\theta}(\tau)$$的定义，个人理解，通过链式法则 $$\pi_ {\theta}$$ 是因为策略而致使当前系统经过序列 $$(s_{1 \cdots \tau},a_{1 \cdots \tau},r_{1 \cdots \tau},)$$ 的概率，

$$
\pi_\theta(\tau) = \mathcal{P}(s_0) \prod_{t=1}^T {\pi_\theta (a_t | s_t) p(s_{t+1},r_{t+1} | s_t, a_t)}
$$

对其进行分解 ，$$\mathcal {P}$$ 表示从某状态 $$s_0$$ 开始遍历的分布。 从那时起，我们应用概率的乘积规则，因为每个新的动作概率都独立于前一个（马尔科夫性）。 在每一步中，我们都会使用 $$\pi_ \theta$$采取一些行为，而环境动态 $$p$$ 决定了因为行为而到达的新状态。 等效地，通过 $$\log$$ 计算，

$$
\begin{aligned} \log \pi_\theta(\tau) &= \log \mathcal{P}(s_0) + \sum_{t=1}^T \log \pi_\theta (a_t | s_t) + \sum_{t=1}^T \log p(s_{t+1},r_{t+1} | s_t, a_t) \\ \nabla \log \pi_\theta(\tau) &= \sum_{t=1}^T \nabla \log \pi_\theta (a_t | s_t) \\ \implies \nabla \mathbb{E}_{\pi_\theta} \left[ r(\tau) \right] &= \mathbb{E}_{\pi_\theta} \left[ r(\tau) \left( \sum_{t=1}^T \nabla \log \pi_\theta (a_t | s_t) \right) \right] \end{aligned}
$$

这里，注意 $$\log p(s_{t+1},r_{t+1} | s_t, a_t)$$ 是一个与时间无关的量，梯度为0。这个结果本身很漂亮，因为它告诉我们，我们真的**不需要知道** $$\mathcal {P}$$ **的遍历分布或环境动态** $$p$$ 。这是至关重要的，因为对于大多数实际目的而言，它很难为这两个变量建模。摆脱它们，无疑是一个进步。因为我们不对环境进行“建模”，所以所产生的算法都称为“无模型算法”，。

不过，我们仍然需要计算“期望”。一种简单但有效的方法是对**大量轨迹**进行**采样**并将其平均。这是一个近似却无偏的估计值。类似于使用离散点集对连续空间上的积分进行近似。该技术的正式名称是马尔可夫链蒙特卡罗（MCMC），MCMC广泛用于概率图形模型和贝叶斯网络中，用于近似参数概率分布。

在我们上面的处理中，一个没有讨论的项是轨迹的奖励 $$r(\tau)$$ 。即使参数化策略的梯度不取决于奖励，该项也会导致MCMC采样后方差的增加。实际上，每个 $$R_t$$ 都有 $$T$$ 个源来增加方差。因为从优化RL目标的角度来看，过去的回报无济于事，所以我们可以改用折现回报 $$G_t$$来代替$$r(\tau)$$。这得出了REINFORCE \[3\]经典策略梯度算法。在进一步讨论后，我们就会发现REINFORNCE并这不能完全解决问题。

### REINFORCE算法（和基线） <a id="reinforce-and-baseline"></a>

重申一下，REINFORCE算法通过如下算法计算策略梯度：

$$
\begin{aligned} \nabla \mathbb{E}_{\pi_\theta} \left[ r(\tau) \right] &= \mathbb{E}_{\pi_\theta} \left[ \left( \sum_{t=1}^T G_t \nabla \log \pi_\theta (a_t | s_t) \right) \right] \end{aligned}
$$

我们仍然没有解决采样轨迹的方差问题。 解决此问题的一种方法是重新定义RL对象，将其定义为似然最大化（Maximum Likelihood Estimate）。 在MLE设置中，众所周知，数据使先验不堪重负-用简单的话来说，无论初始估计有多差，在数据限制内，模型都会收敛到真实参数。 但是，在数据样本具有高方差的情况下，稳定模型参数可能会非常困难。 在我们的上下文中，任何不稳定的轨迹都可能导致政策分配出现次优的转变。 回报规模加剧了这个问题。

因此，我们改为尝试引入另一个称为基线 $$b$$ 的变量来优化报酬差异。 为了使梯度估计保持不偏不倚，基线与策略参数无关。

有基准线的REINFORCE：

$$
\begin{aligned} \nabla \mathbb{E}_{\pi_\theta} \left[ r(\tau) \right] &= \mathbb{E}_{\pi_\theta} \left[ \left( \sum_{t=1}^T (G_t - b) \nabla \log \pi_\theta (a_t | s_t) \right) \right] \end{aligned}
$$

要了解原因，我们必须证明梯度与附加项保持不变。

$$
\begin{aligned} \mathbb{E}_{\pi_\theta} \left[ \left( \sum_{t=1}^T b \nabla \log \pi_\theta (a_t | s_t) \right) \right] &= \int \sum_{t=1}^T \pi_\theta (a_t | s_t) b \nabla \log \pi_\theta (a_t | s_t) d\tau \\ &= \int \sum_{t=1}^T \nabla b \pi_\theta (a_t | s_t) d\tau \\ &= \int \nabla b \pi_\theta (\tau) d\tau \\ &= b \nabla \int \pi_\theta (\tau) d\tau \\ &= b \nabla 1 \\ \mathbb{E}_{\pi_\theta} \left[ \left( \sum_{t=1}^T b \nabla \log \pi_\theta (a_t | s_t) \right) \right] &= 0 \end{aligned}
$$

在理论和实践中，使用基线都可以减少方差，同时保持梯度不变。 一个好的基准将是使用状态值当前状态。

状态值\[8\]：状态值定义为在遵循策略 $$\pi_ \theta$$ 的状态下的预期收益。

$$
V(s) = \mathbb{E}_{\pi_\theta}[G_t | S_t = s]
$$

## Actor-Critic Methods

寻找一个好的基线本身就是很有挑战性的，计算同样也是一个难题。 换一个角度，我们也用参数 $$\omega$$ 的近似来估计 $$V ^\omega(s)$$ 。 我们使用可学习$$V ^\omega(s)$$的梯度的算法被称为Actor-Critic Algorithms \[4，6\]，因为此值函数类似于为“ actor”（个体策略）估计一个“ critic”（值（好坏）。 这一次，我们须计算actor员和critic的梯度。

单步自举回报：单步自举回报获得**即时回报**，并通过使用轨迹中下一个状态的**自举价值**估计来估算回报。

$$
G_t \simeq R_{t+1} + \gamma V^\omega(S_{t+1})
$$

Actor-Critic梯度更新为（个人理解，对于Actor来说）：

$$
\begin{aligned} \nabla \mathbb{E}_{\pi_\theta} \left[ r(\tau) \right] &= \mathbb{E}_{\pi_\theta} \left[ \left( \sum_{t=1}^T (R_{t+1} + \gamma V^\omega(S_{t+1}) - V^\omega(S_{t})) \nabla \log \pi_\theta (a_t | s_t) \right) \right] \end{aligned}
$$

不用说，我们还需要更新critic的参数 $$\omega$$ 。 通常认为目标函数是均方损耗（或更软性的Huber损耗），以及可以使用随机梯度下降法更新参数。

critic的目标函数是：

$$
\begin{aligned} J(\omega) &= \frac{1}{2}\left(R_{t+1} + \gamma V^\omega(S_{t+1}) - V^\omega(S_{t})\right)^2 \\ \nabla J(\omega) &= R_{t+1} + \gamma V^\omega(S_{t+1}) - V^\omega(S_{t}) \end{aligned}
$$

## Generic Reinforcement Learning Framework

现在，我们可以得出一种通用算法，以了解我们学到的所有部分在哪里融合在一起。 所有新算法通常都是下面给出的算法的一种变体，试图攻击一个（或问题的多个步骤）。

```text
Loop:
    Collect trajectories (transitions - (state, action, reward, next state, terminated flag))
    (Optionally) store trajectories in a replay buffer for sampling
    Loop:
        Sample a mini batch of transitions
        Compute Policy Gradient
        (Optionally) Compute Critic Gradient
        Update parameters
```

对于熟悉Python的读者而言，这些代码段旨在更直观地表示上述理论思想。 这些已从实际代码的学习循环中删除。

### Policy Gradients \(Synchronous Actor-Critic\)

```text
# Compute Values and Probability Distribution
values, prob = self.ac_net(obs_tensor)
# Compute Policy Gradient (Log probability x Action value)
advantages = return_tensor - values
action_log_probs = prob.log().gather(1, action_tensor)
actor_loss = -(advantages.detach() * action_log_probs).mean()
# Compute L2 loss for values
critic_loss = advantages.pow(2).mean()
# Backward Pass
loss = actor_loss + critic_loss
loss.backward()
```

### Deep Deterministic Policy Gradients

```text
# Get Q-values for actions from trajectory
current_q = self.critic(obs_tensor, action_tensor)
# Get target Q-values
target_q = reward_tensor + self.gamma * self.target_critic(next_obs_tensor, self.target_actor(next_obs_tensor))
# L2 loss for the difference
critic_loss = F.mse_loss(current_q, target_q)
critic_loss.backward()
# Actor loss based on the deterministic action policy
actor_loss = - self.critic(obs_tensor, self.actor(obs_tensor)).mean()
actor_loss.backward()
```

## References

1. Bertsekas, D.P. et al., 1995. Dynamic programming and optimal control, Athena scientific Belmont, MA.
2. Lillicrap, T.P. et al., 2015. Continuous control with deep reinforcement learning. CoRR, abs/1509.02971. Available at: http://arxiv.org/abs/1509.02971.
3. Mnih, V. et al., 2016. Asynchronous methods for deep reinforcement learning. In International conference on machine learning. pp. 1928–1937.
4. Silver, D. et al., 2014. Deterministic policy gradient algorithms. In ICML.
5. Sutton, R.S. et al., 2000. Policy gradient methods for reinforcement learning with function approximation. In Advances in neural information processing systems. pp. 1057–1063.
6. Sutton, R.S. & Barto, A.G., 2018. Reinforcement learning: An introduction, MIT press.
7. Watkins, C.J. & Dayan, P., 1992. Q-learning. Machine learning, 8\(3–4\), pp.279–292.
8. Williams, R.J., 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8\(3–4\), pp.229–256.

