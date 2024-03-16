# **这是我的第二篇sci论文以及对应的代码**<br>


**摘要**：随着线上交易的普及，信用卡欺诈事件发生频率也越来越高，而目前信用卡欺诈检测中最常使用自适应增强(adaboost)模型，因此如何提高传统adaboost算法的鲁棒性已然成为一个热点问题。传统adaboost算法之所以鲁棒性较差很大一部分原因是由于基分类器的选取方式是以错误率为唯一导向的，因此本文利用一种自适应混合权重的自步学习方法，改进adaboost算法的目标函数，从而改变了adaboost算法中基学习器选择的策略，同时本文选取的自步学习自带的自适应阈值求取算法可以很好的减轻人为经验对于模型训练的影响。本文还从泛化误差的角度出发，选用双误度量来计算基分类之间多样性程度，在弱学习器的权重计算中加入多样性的影响系数，并通过实验给出影响系数的最佳范围，最后本文将提出的改进算法应用于信用卡欺诈场景中。并且与目前几种卓有成效的adaboost改进算法进行实验比较，实验表明本文提出的改进算法在AUC值和F1值上的综合表现要优于其他算法，由此可以证明本文的改进算法可以有效提高传统adaboost在信用卡欺诈检测的算法性能。<br><br><br>


* artice<br>
对应的三种SPL思路的论文<br>
* train<br>
论文所用到的各算法源码，以及训练集<br>
