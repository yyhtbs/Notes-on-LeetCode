# Alberta University \(Reinforcement Learning\)

老虎机问题是为了对不确定情况下进行决策的建模。

主要包含3个方面：行为，回报，值函数

动作和值：

值的意思是回报的期望：

$$
Q^*(a) \doteq \mathbb{E}[R_t|A_t=a] =\sum_rp(r|a)r \text{ for }\forall a \in {1 \dots k}
$$

K-Armed Bandit问题是最大化目标回报：

$$
\underset{a}\text{argmax }Q^*(a)
$$

 之所以讨论老虎机问题，是因为他是RL中的最简单的设定。

最大化回报和估计Q值是老虎机和RL中的核心子问题。



## 学习行为值函数

### 通过采样平均方法估计行为值函数

采样平均方法说白了就是总回报除以总行为，

$$
Q^*(a)=\frac{\text{sum of rewards when action taken prior to \it{t}}}{\text{number of times action taken prior to \it{t}}}=\frac{\sum^{t-1}_{i=1}R_i}{t-1}
$$

### 描述贪婪行为

通过的之前实验，医生只选择最好的处方（最大的行为值） $$\underset a \text{argmax}Q^*(a)$$ 来治疗病人，这个被称为贪婪法。

### 探测-利用困境

医生可能通过尝试并非最好的办法来治疗病人，这个称之为探索。这时，医生会损失立即回报来获得更多的信息。

困境指的是医生不能同时执行探索和利用。

## 增量估计

考虑一个情况，如果要选择在网页上展示广告。被点击最多的广告是最大回报的。采样平均方法需要记录所有广告的点击信息，在一些场景下这会非常浪费存储。对于一些算法，是可以把他们修改称增量型的。这个增量估计是学习过程中的一种更通用的值估计方法。

比如说求均值的一般形式：

$$
Q_{n+1}=\frac{1}{n}\sum_{i=1}^{n}R_i
$$

这个可以通过变换变成一个迭代形式：

$$
Q_{n+1}=\frac{1}{n}R_n+
\frac{1}{n}\sum_{i=1}^{n-1}R_i=
\frac{1}{n}R_n+\frac{n-1}{n}Q_n=
\frac{1}{n}(R_n+(n-1)Q_n)
$$

通过变形，

$$
\frac{1}{n}(R_n+(n-1)Q_n)=
\frac{1}{n}(R_n+nQ_n-Q_n)=
Q_n+\frac{1}{n}(R_n-Q_n)
$$

这是一个增量更新法则，

$$
\textcolor{blue}{\text{NewEstimate}}\leftarrow \textcolor{red}{\text{OldEstimate}}{+ \textcolor{green}{\text{StepSize}}[Target - \textcolor{red}{\text{OldEstimate}}]}\\
\textcolor{blue}{Q_{n+1}}=\textcolor{red}{Q_n}+\textcolor{green}{\frac{1}{n}}(R_n-\textcolor{red}{Q_n})
$$

这里Target是指立即的获取的值 $$R_n$$ 。从系统稳定的角度考虑，StepSize是一个0到1之间的数。在求均值的情况下，步长是是一个变化的值。

考虑个时变的情况，如果医生药只在固定的时间更有效。这个成为非平稳老虎机问题。回报的分布随着时间的变化而变化。医生不知道这个变化，但是适应这个过程。

一个最简单的方案是用一个固定的步长。

通过把这个公式展开，

$$
Q_{n+1}=Q_n+\alpha(R_n-Q_n)=\alpha R_n+(1-\alpha)\alpha R_{n-1}\\
+(1-\alpha)^2\alpha R_{n-2} \dots (1-\alpha)^{n-1}\alpha R_1
+(1-\alpha)^nQ_1 \\
Q_{n+1}=(1-\alpha)^n Q_1+\sum_{i=1}^n\alpha (1- \alpha)^{n-i}R_i
$$

这个式子表示随着时间的推移，最初的估计对当前的估计影响指数变小。后面一项指的是随着时间的推移，之前获取的回报对当前的估计的贡献也是逐渐减小的，因为 $$(1-\alpha)^n$$ 对 $$n$$ 来说是一个递减函数。

最近的回报对估计影响最大。





