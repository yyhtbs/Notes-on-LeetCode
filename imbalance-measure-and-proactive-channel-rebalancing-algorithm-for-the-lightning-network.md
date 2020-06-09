# ICBC2020闪电网络中通道不平衡度量和主动重平衡算法

通过尝试几种付款路径直到成功，才能在具有隐私意识的支付渠道网络中进行付款。对于大型网络（如闪电网络），完成一次付款可能需要花费几分钟。我们介绍了一种网络不平衡措施，并提出了改善网络平衡的优化问题，作为网络内循环路径上通道内资金的一系列再平衡操作。由于渠道的资金和余额并不为全球所知，因此我们引入了一种贪婪的启发式方法，尽管存在不确定性，但每个节点都可以使用它来改善自身的本地余额。在最近的闪电网络快照的经验模拟中，我们证明了与应用启发式方法后的不平衡分布相比，网络的不平衡分布的Kolmogorov-Smirno距离为0:74。我们进一步证明，单笔付款的成功率从不平衡网络上的11：2％增加到平衡网络上的98：3％。类似地，在最便宜的路径上进行初始路由尝试时，所有参与者对之间的可能的中位数支付大小将从0增加到0.5 mBTC。我们提供的经验证据表明，应该为主动的再平衡操作降低路由费。执行4种选择重新平衡周期的不同策略会导致类似的结果，表明从实际的角度来看，在朋友网络的朋友内部进行协作可能更为可取。

## Formalisation and Assumption

让 $$N = (V, E, c)$$ 是一个有有限节点的支付网络。支付通道是网络的边$$E \subset V \times V$$。对于每一条边，有一个公开的容量函数 $$ c : E  \rightarrow N$$。对于一条边，$$e = (u, v)$$，我们有$$e_u := (e, u)$$ 代表第一个参与者 $$u$$ ；$$e_v = (e, v)$$ 代表第二个参与者。自然的，通道$$e = (u, v)$$ 的容积被由私有的余额函数$$b : E \times V \rightarrow N$$ 组成，满足$$b(e_u) + b(e_v) = c(e)$$ .

我们将信道上对于参与者 $$u$$ 的信道平衡系数定义为 $$ζ(u,v) = \frac{b(e_u)}{c(e)}$$ 。 这只是参与者 $$u$$ 在通道 $$e$$ 中拥有的相对资金量。 由于 $$b(e_u) + b(e_v) = c(e)$$ ，我们也有 $$ ζ(u,v) + ζ(v,u) = 1$$ 。通常，对于不平衡的通道，我们有 $$ζ(u,v) \neq ζ(v,u)$$ 。定义一个邻居函数 $$ n : V \rightarrow 2^E $$ 为每个节点分配它的所有通道集合。 为了使以下公式更易于阅读，让我们介绍邻居集合 $$U := n(u)$$ 。

参与者 $$u$$ 的总资金表示为 $$τ_u := \sum_{e \in U}{b(e_u)}$$ 。当不进行任何付款，也不打开和关闭任何通道时，$$τ_u$$ 的值恒定。 相反，余额函数 $$b$$ 会随着资金的重新分配而变化。 对于参与者 $$u$$ 来说，他的总容量是 $$ κ_u := \sum_{e \in U}{c(e)}$$。使用最后两个定义，让我们可以导出参与者 $$u$$ 的节点平衡系数为 $$v_u= \frac{τ_u}{κ_u}$$ 。

如果节点的信道平衡系数 $$ζ(u,v_1) \dots ζ(u,v_d)$$ 相同，我们称其为 $$u$$ 平衡。这意味着节点在所有通道上的相对资金分配是相同的。 因此，如果节点的本地信道平衡系数不相等，则认为该节点不平衡。 从统计上讲，分布的不平等性可以用基尼系数来衡量。 因此，对于具有信道平衡系数$$ζ(u,v_1) \dots ζ(u,v_d)$$ 

我们定义基尼系数，

$$
G_u=\frac{\sum_{i \in U}\sum_{j \in U}|ζ_i-ζ_j|}{2\sum_{j \in U}ζ_j}
$$

如果 $$G_u = 0$$ ，这意味着信道平衡系数相等。 相反，如果 $$G_u = 1$$ ，则信道平衡系数以最不平等的方式分配。

注意，当且仅当其信道平衡系数全部取相同值时，节点 $$u$$ 的信道平衡系数的基尼系数取值为0。 该值将与节点的平衡系数 $$ν_u$$ 完全相同。 如果我们使用绝对平衡定义 $$ζ$$ 值，容量差异较大的信道的节点将不太可能达到0的基尼系数，因为在重新平衡操作期间，较小的信道可能会被较大的信道耗尽。

最后，我们定义 $$G$$ 为网络的不平衡系数。 $$G=\frac{1}{|V|}\sum_{v \in V} G_v$$ 。它是网络中所有节点的不平衡值的平均值。 如果 $$G=0$$ ，则将实现一个完美的平衡网络，而如果G的值接近1，则平衡将很差。

我们的目标是找到一个平衡函数 $$b$$ ，该函数在给定隐私要求下且支付通道且资金初始分配为 $$\tau _{u_1} \dots \tau _{u_n} $$ ，最小化 $$G$$ 。 此优化问题的约束条件是，对于每个节点 $$u \in V$$ 和平衡函数 $$b$$ 的任何选择，总资金 $$t_u$$ 是固定的。 在具有隐私要求的支付网络中，资金分配$$\tau _{u_1} \dots \tau _{u_n} $$并不是公开的。同样，初始平衡函数也不是公开的。 由于我们缺乏有关全局网络状态的知识，因此无法使用标准的优化技术（例如梯度下降，共轭梯度法或是模拟退火）。我们推荐使用一个启发式方法，使得每一个参与者进行某些操作来提升他的平衡度量。
