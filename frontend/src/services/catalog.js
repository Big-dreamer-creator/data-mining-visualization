export const fallbackAlgorithms = [
  {
    id: "knn",
    name: "K-近邻算法",
    shortName: "KNN",
    summary: "基于空间距离度量的投票分类机制。",
  },
  {
    id: "logistic-regression",
    name: "逻辑回归",
    shortName: "Logistic",
    summary: "用线性边界和概率映射完成分类。",
  },
  {
    id: "lda",
    name: "线性判别分析",
    shortName: "LDA",
    summary: "最大化类间差异、最小化类内差异。",
  },
  {
    id: "decision-tree",
    name: "决策树",
    shortName: "Tree",
    summary: "通过层级规则递归划分特征空间。",
  },
  {
    id: "adaboost",
    name: "AdaBoost 集成学习",
    shortName: "AdaBoost",
    summary: "动态加权多个弱分类器形成强分类器。",
  },
  {
    id: "mlp",
    name: "多层感知机",
    shortName: "MLP",
    summary: "用神经网络学习复杂非线性边界。",
  },
];

export const fallbackDatasets = [
  { id: "iris", name: "Iris 鸢尾花数据集" },
  { id: "wine", name: "Wine 葡萄酒数据集" },
];

export const tabs = [
  { id: "principle", label: "原理介绍" },
  { id: "dataset", label: "数据集" },
  { id: "visualization", label: "执行过程" },
  { id: "analysis", label: "结果分析" },
];
