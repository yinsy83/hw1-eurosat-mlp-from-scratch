# HW1 实验报告模板

## 1. 任务介绍

- 数据集：EuroSAT RGB，共 10 个类别
- 任务目标：使用从零实现的三层神经网络对遥感图像进行土地覆盖分类
- 实现约束：不使用 PyTorch、TensorFlow、JAX 等自动微分框架，仅使用 NumPy 等基础库

## 2. 方法设计

### 2.1 数据预处理

- 图像大小：`64 x 64 x 3`
- 输入表示：展平为一维向量
- 数据划分：训练集 / 验证集 / 测试集
- 归一化方式：仅使用训练集统计均值和标准差

### 2.2 模型结构

- 模型结构：`Input -> Hidden1 -> Hidden2 -> Output`
- 隐藏层维度：`[256, 128]` 或实验最优配置
- 激活函数：`ReLU / Tanh / Sigmoid`
- 输出层：类别 logits

### 2.3 自动微分与反向传播

- 自定义 `Tensor` 类，保存：
  - 数据
  - 梯度
  - 父节点
  - 反向传播函数
- 支持的运算：
  - 加减乘除
  - 矩阵乘法
  - reshape / transpose
  - sum / mean
  - ReLU / Sigmoid / Tanh
- 损失函数：带稳定化处理的 Softmax Cross-Entropy
- 正则项：L2 正则化

### 2.4 优化方法

- 优化器：SGD
- 学习率衰减：Step Decay
- 最优模型选择：依据验证集 Accuracy 保存最佳权重

## 3. 实验设置

### 3.1 超参数范围

可给出表格：

| 参数 | 备选值 |
|---|---|
| learning rate | 0.05, 0.01 |
| hidden dims | [256,128], [512,256] |
| weight decay | 1e-4, 1e-3 |
| activation | ReLU, Tanh |
| batch size | 64 |

### 3.2 训练配置

- epoch 数
- batch size
- 学习率衰减步长
- 随机种子

## 4. 实验结果

### 4.1 超参数搜索结果

插入 `outputs/search/grid_search_results.json` 中整理出的表格。

### 4.2 训练曲线

插入：

- `outputs/curves/*_curves.png`

需要分析：

- 训练集和验证集 Loss 是否同步下降
- 是否存在过拟合或欠拟合
- 验证集 Accuracy 的变化趋势

### 4.3 测试集结果

- 最终测试准确率
- 混淆矩阵图
- 各类别分类效果差异分析

## 5. 第一层权重可视化分析

插入：

- `outputs/weight_viz/*_first_layer_weights.png`

建议分析角度：

- 是否出现针对颜色的偏好，如蓝色、水体、绿色植被
- 是否出现条带状、块状或边缘结构
- 哪些隐藏单元似乎更偏向纹理模式，哪些更偏向颜色模式

## 6. 错例分析

插入：

- `outputs/error_cases/*_error_cases.png`

分析建议：

- `River` 与 `Highway` 在细长带状结构上可能相似
- `Pasture` 与 `HerbaceousVegetation` 在纹理和颜色上接近
- `Residential` 与 `Industrial` 在建筑分布上有重叠
- 部分样本存在多类地物混杂，导致边界模糊

## 7. 总结

- 从零实现自动微分和三层 MLP 的主要收获
- 模型在 EuroSAT 上的表现
- 模型局限性：MLP 缺乏卷积结构，对空间局部模式利用不如 CNN
- 后续改进方向：更系统的超参数搜索、更好的输入预处理、更深层 MLP 等

## 8. 提交材料检查表

- 代码已上传至 Public GitHub Repo
- README 包含环境依赖与运行说明
- 报告中包含 GitHub Repo 链接
- 报告中包含模型权重下载链接
- 链接更新时间早于 deadline
