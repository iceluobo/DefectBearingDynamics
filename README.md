# 轴承动力学求解
## 滚子-内外滚道刚度求解
KeyStiffnesses.py 输出轴承设计参数及材料参数，输出滚动体和内/外滚道之
间的非线性接触刚度Ki，Ko，输出刚度拟合曲线pdf，并保存至对应的txt文件。

注意检查函数输出和txt文件匹配。
## 轴承动力学模型方程组
BearingModel.py 轴承动力学模型方程组函数及接触力求解函数
### ode_system
在 main.py 中使用 solve_ivp 求解该微分方程组

需要参数：

1. Fr：内圈径向外载荷
2. Nb：滚子数目
3. mi：内圈质量
4. mo：外圈质量
5. wc：保持架理论转速
6. Ki：滚子-内滚道接触刚度
7. Ko：滚子-外滚道接触刚度
8. Kbh：外滚道接地刚度
9. Cbh：外滚道接地阻尼

其中，Ki，Ko 可以不作区分。
### contact_force
通过内外滚道的位移差求解滚道所受的非线性力总和。
### delta_calculate
通过内外滚道的位移求解每个滚子位置的相对压缩量，在 contact_force 中进一步转化为非线性接触力。

## 模型求解及数据保存
main.py 中进行模型求解及数据保存
### healthy_bearing
使用 solve_ivp 求解健康轴承的动力学微分方程组。
### load_K
导入 KeyStiffnesses.py 计算得到的拟合接触刚度。