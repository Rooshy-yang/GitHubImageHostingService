<img src="https://raw.githubusercontent.com/rooshy-yang/GitHubimageHostingService/master/img/RIG/image-20220608163321876.png" alt="image-20220608163321876" style="zoom:67%;" />

该论文提出一种基于目标的强化学习**goal-conditioned** reinfocement learning，并利用VAE的生成模型得到相应的goal分布，然后根据分布采样出特定的goal设定内在奖励来提升训练的效率。

为什么使用VAE有如下三个原因：

首先，环境中的状态和观察可能并不是一一对应的，而利用VAE来训练得到的编码器可能有关于特定观察下更本质的特征，即使在现实世界中也可以从像图片这样的观察中学习。

其次，它允许对新状态进行采样，这些状态可用于在训练期间设置综合目标，以允许目标条件策略练习不同的行为。我们还可以通过在off-policy的RL算法中重新标记(relabel 技术也是一个贡献之一)合成目标来更有效地利用环境中的样本，使我们的算法在训练中更加高效。

第三，VAE的学习表示提供了一个空间，其中距离比原始观测空间更有意义，因此可以为RL提供形状良好的奖励函数。通过学习实现从潜在变量模型中抽样的随机目标，目标条件策略可以了解世界，并可用于在测试时实现新的、指定的目标。论文中使用的编码器符合高斯分布 <img src="https://www.zhihu.com/equation?tex=q_φ(s) = N(μ_φ(s),σ_φ^2(s))" alt="q_φ(s) = N(μ_φ(s),σ_φ^2(s))" class="ee_img tr_noresize" eeimg="1"> ，所以奖励设置为：

<img src="https://raw.githubusercontent.com/rooshy-yang/GitHubimageHostingService/master/img/RIG/image-20220608160212563.png" alt="image-20220608160212563" style="zoom:50%;" />

其中 <img src="https://www.zhihu.com/equation?tex=z_g " alt="z_g " class="ee_img tr_noresize" eeimg="1"> 表示利用VAE编码器 <img src="https://www.zhihu.com/equation?tex=e" alt="e" class="ee_img tr_noresize" eeimg="1"> 生成的隐(latent)目标  <img src="https://www.zhihu.com/equation?tex=z_g = e(g)" alt="z_g = e(g)" class="ee_img tr_noresize" eeimg="1"> 。  <img src="https://www.zhihu.com/equation?tex=|| z - z_g ||_A" alt="|| z - z_g ||_A" class="ee_img tr_noresize" eeimg="1"> 是关于隐空间中的马氏距离，A为VAE编码器的Precision matrix。

我们把探索的状态的作为初始goal，这样的好处是有很好的普遍性。下一步，为了训练VAE，通过执行随机策略和收集状态观察 <img src="https://www.zhihu.com/equation?tex=\{s_i\}" alt="\{s_i\}" class="ee_img tr_noresize" eeimg="1"> 来训练  <img src="https://www.zhihu.com/equation?tex=\beta-VAE" alt="\beta-VAE" class="ee_img tr_noresize" eeimg="1"> ，对于VAE，我们最大化下面式子：

<img src="https://raw.githubusercontent.com/rooshy-yang/GitHubimageHostingService/master/img/RIG/image-20220608161242924.png" alt="image-20220608161242924" style="zoom:50%;" />

其中 <img src="https://www.zhihu.com/equation?tex=p(z)" alt="p(z)" class="ee_img tr_noresize" eeimg="1"> 是某种先验，可以取正态分布。 <img src="https://www.zhihu.com/equation?tex=\beta" alt="\beta" class="ee_img tr_noresize" eeimg="1"> 是一个超参数用来平衡两个项的比重。编码器参数化高斯分布的均值和对数方差对角线。而解码器 <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> 参数化每个像素值的伯努利分布。

整个算法流程如下：

<img src="https://raw.githubusercontent.com/rooshy-yang/GitHubimageHostingService/master/img/RIG/image-20220608162438753.png" alt="image-20220608162438753" style="zoom:67%;" />

该论文直接使用了TD3 作为训练模型，优化贝尔曼误差：

<img src="https://raw.githubusercontent.com/rooshy-yang/GitHubimageHostingService/master/img/RIG/image-20220608162703454.png" alt="image-20220608162703454" style="zoom: 33%;" />



相关代码：https://github.com/vitchyr/rlkit.

相关论文：https://arxiv.org/abs/1807.04742
