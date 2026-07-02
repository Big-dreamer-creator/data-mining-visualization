# 数据挖掘与机器学习分类算法可视化系统

本项目是一个基于前后端分离架构的分类算法教学与演示 Demo。系统旨在通过直观的 Web 界面，展示从经典数据挖掘到现代深度学习分类算法的核心原理和执行过程。

系统采用“状态帧驱动”架构：前端只提交数据集、二维特征索引和超参数；FastAPI 后端一次性完成模型训练、几何计算、网格预测和辅助线计算，并返回完整状态帧数组。前端只维护当前步骤索引，点击下一步时播放对应状态帧。

## ✨ 核心功能模块 (TODO)

系统主要包含以下六大经典分类算法的运行原理与可视化展示：

### 1. 传统统计与距离模型
- [x] **K-近邻算法 (KNN)**
  - **核心原理**：基于空间距离度量的投票分类机制。
  - **可视化展示**：未知样本到所有已知样本的距离连线、最近 K 个邻居高亮、投票结果和网格决策背景。
- [x] **逻辑回归 (Logistic Regression)**
  - **核心原理**：基于 Sigmoid 函数的线性边界概率映射。
  - **可视化展示**：随机初始边界、梯度下降边界演变、样本到边界误差垂线和最终线性决策区域。
- [x] **线性判别分析 (Linear Discriminant Analysis - LDA)**
  - **核心原理**：“类内方差最小，类间方差最大”的投影降维与分类思想。
  - **可视化展示**：类别中心、最优投影轴、正交投影点、投影虚线、轴上阈值和最终线性背景。

### 2. 树模型、集成学习与深度学习
- [x] **决策树 (Decision Tree)**
  - **核心原理**：基于信息增益或基尼系数的特征空间层级划分。
  - **可视化展示**：按节点展开每一次水平或垂直切分，最终形成正交矩形决策区域。
- [x] **AdaBoost 集成学习**
  - **核心原理**：通过前向分布算法将多个弱分类器动态加权聚合成强分类器。
  - **可视化展示**：每一轮迭代中错分样本权重 (Sample Weights) 放大的视觉变化追踪。
- [x] **多层感知机 (Multilayer Perceptron - MLP)**
  - **核心原理**：基于反向传播算法和激活函数的前馈神经网络模型。
  - **可视化展示**：不同 Epoch 的模型快照，以及隐藏层如何逐步扭曲空间生成非线性决策边界。

## 🛠️ 技术栈选型

- **前端 (Frontend)**: Vue 3 + Vite + ECharts (用于高质量数据图表与边界动态渲染)
- **后端 (Backend)**: FastAPI (提供轻量、高性能的异步 RESTful API)
- **核心算法库**: 
  - `Scikit-learn` (封装核心算法与模型评估工具)
  - `NumPy` & `Pandas` (用于高效矩阵运算与数据集预处理)

## 📁 目录结构说明

- `/frontend`: Vue 3 前端工程代码
- `/backend`: FastAPI 后端工程代码
  - `/api`: RESTful 路由与接口控制器
  - `/core`: 算法模型封装与核心运算逻辑
- `/dataset`: 存放系统运行所需的测试数据集 (如 CSV 文件)

## 🚀 启动指南

### 后端：uv + 项目虚拟环境

```bash
cd backend
uv venv
uv pip install -r requirements.txt
.venv\Scripts\python scripts\download_datasets.py
.venv\Scripts\uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

后端接口文档启动后访问：

```text
http://localhost:8000/docs
```

### 前端：bun

```bash
cd frontend
bun install
bun run dev
```

前端默认访问：

```text
http://localhost:5173
```

### 数据集

项目使用两个经典公开数据集，并保存到根目录 `dataset/`：

- `dataset/iris.csv`: Iris 鸢尾花数据集
- `dataset/wine.csv`: Wine 葡萄酒数据集

这些数据集来自 UCI Machine Learning Repository，并通过 `scikit-learn` 内置数据集接口写入本地 CSV。

### API 概览

- `GET /api/health`: 健康检查
- `GET /api/algorithms`: 获取算法列表
- `GET /api/algorithms/{algorithm_id}`: 获取算法说明
- `GET /api/datasets`: 获取数据集列表
- `GET /api/datasets/{dataset_id}`: 获取数据集详情
- `POST /api/algorithms/{algorithm_id}/run`: 运行指定算法并返回完整执行过程状态帧

运行接口请求体：

```json
{
  "datasetId": "iris",
  "featureIndices": [2, 3],
  "hyperparameters": {}
}
```

运行接口响应核心结构：

```json
{
  "algorithmId": "knn",
  "algorithmName": "K-近邻算法",
  "dataset": {},
  "labels": [],
  "coordinateSystem": {},
  "parameters": {},
  "steps": [
    {
      "index": 0,
      "description": "当前算法执行步骤说明",
      "scatter": [],
      "backgroundGrid": null,
      "helpers": [],
      "annotations": []
    }
  ]
}
```
