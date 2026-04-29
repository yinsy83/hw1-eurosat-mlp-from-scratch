# 最终结果摘要

## 最终选定模型

- 配置：`ReLU + [512, 256] + lr=0.01 + weight_decay=1e-3`
- 验证集最佳准确率：`0.6582`
- 测试集准确率：`0.6614`
- 权重文件：`outputs/checkpoints/compare_relu_512_256_wd1e3_best.npz`

## 关键产物

- 训练曲线：`outputs/curves/compare_relu_512_256_wd1e3_curves.png`
- 训练历史：`outputs/curves/compare_relu_512_256_wd1e3_history.json`
- 混淆矩阵：`outputs/confusion_matrix/compare_relu_512_256_wd1e3_test_cm.png`
- 各类准确率：`outputs/confusion_matrix/compare_relu_512_256_wd1e3_per_class_acc.png`
- 错例分析：`outputs/error_cases/compare_relu_512_256_wd1e3_error_cases.png`
- 第一层权重可视化：`outputs/weight_viz/compare_relu_512_256_wd1e3_first_layer_weights.png`
- 超参数搜索：`outputs/search/grid_search_results.json`

## 备用结果

- 配置：`ReLU + [256, 128] + lr=0.01 + weight_decay=1e-4`
- 验证集最佳准确率：`0.6471`
- 测试集准确率：`0.6629`
- 权重文件：`outputs/checkpoints/final_best_relu_256_128_best.npz`

## 仍需你手动补的内容

- GitHub Public Repo 链接
- 模型权重网盘下载链接
- 将 `report/HW1_report.md` 导出为 PDF
