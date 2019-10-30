## 哈尔滨工业大学2019年秋季学期机器学习课程实验一

### **实验名称**：多项式拟合正弦曲线

### **目标**：

- 掌握最小二乘法求解（无惩罚项的损失函数）、掌握加惩罚项（2范数）的损失函数优化、梯度下降法、共轭梯度法、理解过拟合、克服过拟合的方法(如加惩罚项、增加样本) 


### **要求**：

- 生成数据，加入噪声；
- 用高阶多项式函数拟合曲线；
- 用解析解求解两种loss的最优解（无正则项和有正则项）
- 优化方法求解最优解（梯度下降，共轭梯度）；
- 用你得到的实验数据，解释过拟合。
- 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。
- 语言不限，可以用matlab，python。求解解析解时可以利用现成的矩阵求逆。梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例如pytorch，tensorflow的自动微分工具。

### **代码说明：**

- 本次实验主要部分采用C++完成。

- draw.py用于画图可视化拟合曲线，依赖于matplotlib和numpy。

- 代码组织架构有些混乱，头文件“链式”引用。

- 具体依赖关系见代码。

- 代码中含有相关注释，运行时只需运行main.cpp即可。

- 项目依赖于第三方矩阵库Eigen，已放入仓库中。

- 有任何问题欢迎私信联系。

- 欢迎阅读[我的博客](https://www.fets.xyz/)，里面有关于最小二乘法的总结。

- 下附一个效果比较好的拟合曲线。


![avatar](https://s2.ax1x.com/2019/09/22/upfwEd.png)