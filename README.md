# HW1: EuroSAT 三层神经网络分类器

本项目使用 `NumPy` 从零实现三层多层感知机（MLP）分类器，并在 `EuroSAT_RGB` 遥感图像数据集上完成土地覆盖分类任务。项目不依赖 PyTorch、TensorFlow、JAX 等自动微分框架，核心训练流程、自动微分、反向传播、SGD 优化、学习率衰减、交叉熵损失和 L2 正则化均为手工实现。

## 项目特点

- 自定义 `Tensor` 自动微分引擎，支持矩阵乘法、逐元素运算、广播求梯度、ReLU/Sigmoid/Tanh
- 自定义三层 MLP：`Input -> Hidden1 -> Hidden2 -> Output`
- 支持验证集选优并自动保存最佳权重
- 支持网格搜索超参数
- 支持测试集准确率、混淆矩阵、分类错例分析
- 支持第一层权重可视化，便于观察空间纹理和颜色模式

## 环境依赖

```bash
pip install -r requirements.txt
```

## 数据集

将老师提供的数据集放在项目根目录下：

```text
hw1/
├─ EuroSAT_RGB/
│  ├─ AnnualCrop/
│  ├─ Forest/
│  ├─ HerbaceousVegetation/
│  ├─ Highway/
│  ├─ Industrial/
│  ├─ Pasture/
│  ├─ PermanentCrop/
│  ├─ Residential/
│  ├─ River/
│  └─ SeaLake/
```

## 目录结构

```text
hw1/
├─ EuroSAT_RGB/
├─ src/
│  ├─ autograd.py
│  ├─ data.py
│  ├─ losses.py
│  ├─ metrics.py
│  ├─ model.py
│  ├─ optim.py
│  ├─ search.py
│  ├─ test.py
│  ├─ train.py
│  ├─ utils.py
│  └─ visualize.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ confusion_matrix/
│  ├─ curves/
│  ├─ error_cases/
│  ├─ search/
│  └─ weight_viz/
├─ report/
├─ README.md
└─ requirements.txt
```

## 训练

示例：训练一个高分版三层 MLP（两个隐藏层）

```bash
python src/train.py ^
  --dataset-root EuroSAT_RGB ^
  --hidden-dims 256,128 ^
  --activation relu ^
  --epochs 30 ^
  --batch-size 64 ^
  --lr 0.05 ^
  --weight-decay 1e-4 ^
  --lr-decay-gamma 0.5 ^
  --lr-decay-step 10
```

训练输出：

- `outputs/checkpoints/*.npz`：最佳模型权重
- `outputs/curves/*_curves.png`：训练/验证 loss 与验证 accuracy 曲线
- `outputs/curves/*_history.json`：训练历史记录

## 超参数搜索

```bash
python src/search.py ^
  --dataset-root EuroSAT_RGB ^
  --epochs 12 ^
  --batch-size 64 ^
  --lrs 0.01,0.005 ^
  --weight-decays 0.0001,0.001 ^
  --activations relu,tanh ^
  --hidden-grid 256,128;512,256 ^
  --lr-decay-step 6
```

搜索结果保存在：

- `outputs/search/grid_search_results.json`

也可以直接运行：

```bash
powershell -ExecutionPolicy Bypass -File .\run_full_experiment.ps1
```

## 测试与可视化

将训练得到的最佳模型路径替换到下面命令：

```bash
python src/test.py ^
  --dataset-root EuroSAT_RGB ^
  --checkpoint outputs/checkpoints/your_best_model.npz
```

测试输出：

- `outputs/confusion_matrix/*_test_cm.png`
- `outputs/confusion_matrix/*_per_class_acc.png`
- `outputs/weight_viz/*_first_layer_weights.png`
- `outputs/error_cases/*_error_cases.png`

## 当前最佳结果

基于当前已完成实验，推荐最终模型为：

- hidden dims：`512,256`
- activation：`relu`
- lr：`0.01`
- weight decay：`1e-3`
- epochs：`12`
- best validation accuracy：`0.6582`
- test accuracy：`0.6614`

对应权重文件：

- `outputs/checkpoints/compare_relu_512_256_wd1e3_best.npz`

## 实验报告建议内容

- 数据集介绍与预处理方式
- 三层 MLP 结构设计
- 自动微分与反向传播实现思路
- 训练设置：批大小、学习率、学习率衰减、L2 正则
- 超参数搜索结果表格
- 训练集/验证集 Loss 曲线
- 验证集 Accuracy 曲线
- 测试集准确率与混淆矩阵
- 第一层权重可视化及空间模式讨论
- 错例分析

## 注意事项

- 本项目默认采用分层划分训练集、验证集、测试集
- 归一化均值和方差仅使用训练集统计量
- 测试脚本使用训练阶段保存的 split 索引和归一化参数，保证复现一致
- 如果只想快速调试，可以在训练或搜索脚本里增加 `--sample-limit-per-class`
