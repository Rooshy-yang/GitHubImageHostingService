vaild action have  Structure  

obseration have his own Structure



VAE 复习：

Auto-Encoding Variational Bayes : 旨在构建一个从隐变量Z到生成目标数据X的模型。







问： VAE和GAN的区别

GAN是利用一个生成器和一个判别器，判别器用于训练出一个生成的数据和真实数据区别的模型。



根据两个分布的直方图可以直接计算kl距离



关于random shoot :，以及一些其他方法？

已知一个分布来进行采样。



Goal 和 Task：















ADAPTIVE PROCEDURAL TASK GENERATION FOR HARD-EXPLORATION PROBLEMS

Automatic Goal Generation for Reinforcement Learning Agents： 利用GAN来生成一系列的goal给智能体来进行训练。



直接把状态观察空间当作目标

每一帧当作一个状态来训练智能体



智能体的状态， LSTM

可以设定一些任务给智能体，然后让智能体去探索/完成任务（根据任务设定奖励），根据智能体去环境改变的情况来改进智能体的目标。



试错

观察，观察只有一部分的信息，他对应的模块，一开始所有的东西没有任何区别

生成一些目标，然后根据Goal来跑模型,

探索效率，提供智能体很多普适性的有价值的信息给智能体。