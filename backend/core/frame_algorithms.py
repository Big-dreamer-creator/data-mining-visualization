from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 仅保留蓝红二分类经典色板
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

ALGORITHMS = {
    "knn": {
        "id": "knn",
        "name": "K-近邻算法",
        "short_name": "KNN",
        "category": "传统统计与距离模型",
        "summary": "基于空间距离度量，让邻近样本通过多数投票决定未知样本类别。",
        "principle": [
            {
                "title": "核心思想",
                "body": "KNN 是一种基于实例的分类方法。训练阶段几乎不学习显式参数，而是保存训练样本；预测时计算未知点到所有已知点的距离，再从最近的 K 个样本中做多数投票。",
            },
            {
                "title": "执行过程",
                "body": "完整过程可以拆成三步：先度量未知点到所有样本的距离，再按距离排序选出 K 个邻居，最后统计邻居类别票数并给出预测类别。",
            },
            {
                "title": "几何含义",
                "body": "KNN 的边界来自样本点之间的局部竞争。K 较小时边界容易破碎，能贴近局部结构；K 较大时边界更平滑，但可能吞掉少数类的小区域。",
            },
            {
                "title": "关键参数",
                "body": "K 值决定邻域大小，距离度量决定相似性的定义。由于距离会受量纲影响，真实数据评价前必须标准化，否则数值范围大的特征会支配邻居排序。",
            },
            {
                "title": "观察重点",
                "body": "重点观察未知点附近是否类别混杂、最近几个邻居距离是否接近、改变 K 值后投票是否翻转。若结果很敏感，通常说明样本位于边界附近或局部样本密度不足。",
            },
        ],
        "deepDive": [
            {
                "title": "为什么 KNN 几乎不训练",
                "body": "KNN 的知识直接存放在样本集合里，所谓训练更多是记忆数据。它把“相似样本有相似标签”作为核心假设，因此预测成本被推迟到了查询时刻。",
            },
            {
                "title": "K 值的偏差与方差",
                "body": "K 很小时模型方差大，容易被局部噪声影响；K 很大时偏差大，边界会被多数区域拉平。调 K 本质上是在局部敏感和整体稳定之间折中。",
            },
            {
                "title": "距离度量的含义",
                "body": "欧氏距离强调直线几何距离，曼哈顿距离强调坐标轴方向累计差异。不同距离会改变邻居排序，因此同一个点可能得到不同投票结果。",
            },
            {
                "title": "高维问题",
                "body": "维度升高后样本之间的距离会变得更接近，最近邻不再那么“近”。这就是 KNN 在高维数据上常遇到的维度灾难。",
            },
            {
                "title": "结果怎么读",
                "body": "如果混淆集中在某两个类别，通常说明它们在特征空间里互相靠近；如果多数错误来自低置信区域，则说明样本位于边界或局部密度不足。",
            },
        ],
    },
    "logistic-regression": {
        "id": "logistic-regression",
        "name": "逻辑回归",
        "short_name": "Logistic",
        "category": "传统统计与距离模型",
        "summary": "通过梯度下降调整线性分类边界，并用 Softmax 输出类别概率。",
        "principle": [
            {
                "title": "核心思想",
                "body": "逻辑回归用线性函数计算类别得分，再通过 Sigmoid 或 Softmax 转成概率。它名字里有“回归”，但在分类任务中学习的是一条或多条线性决策边界。",
            },
            {
                "title": "训练目标",
                "body": "模型通过最小化交叉熵损失来调整权重。预测错误或置信度不足时损失更大，梯度下降会推动边界向能降低总体损失的方向平移和旋转。",
            },
            {
                "title": "概率解释",
                "body": "边界两侧不是简单的硬切分，而是概率从一类逐渐过渡到另一类。离边界越远，模型通常越自信；靠近边界的样本更容易成为误判点。",
            },
            {
                "title": "正则化作用",
                "body": "L2 正则会限制权重过大，使边界更稳定。正则太弱可能过度追逐训练集，正则太强则可能欠拟合，边界会显得过于保守。",
            },
            {
                "title": "观察重点",
                "body": "重点观察初始随机边界、误差垂线、梯度下降中边界如何移动，以及最终线性区域是否能解释样本分布。如果数据本身呈环形或月牙形，线性边界会天然吃力。",
            },
        ],
        "deepDive": [
            {
                "title": "线性得分到概率",
                "body": "逻辑回归先计算权重与特征的线性组合，再把得分压缩为概率。二分类常用 Sigmoid，多分类常用 Softmax。",
            },
            {
                "title": "交叉熵为什么合适",
                "body": "交叉熵会惩罚“错得很自信”的预测。模型不只是要预测正确类别，还要让正确类别的概率尽可能高。",
            },
            {
                "title": "边界为什么是直线",
                "body": "当两个类别概率相等时，线性得分相等。这个等式在二维里对应一条直线，在高维里对应一个超平面。",
            },
            {
                "title": "正则化的直觉",
                "body": "正则化会惩罚过大的权重，让模型不要为了少数训练样本把边界推得过激。它牺牲一点训练拟合，换取更稳定的泛化。",
            },
            {
                "title": "什么时候不适合",
                "body": "如果数据呈环形、月牙形或多块区域交错，线性边界无法自然表达结构。此时继续调学习率和迭代次数也只能有限改善。",
            },
        ],
    },
    "lda": {
        "id": "lda",
        "name": "线性判别分析",
        "short_name": "LDA",
        "category": "传统统计与距离模型",
        "summary": "寻找能最大化类间距离并最小化类内散布的一维判别轴。",
        "principle": [
            {
                "title": "核心思想",
                "body": "LDA 寻找一个判别方向，让同类样本投影后尽量集中，不同类别的投影中心尽量分开。它关心的是分类可分性，而不是单纯保留最大方差。",
            },
            {
                "title": "统计假设",
                "body": "LDA 会估计各类别均值和类内散布，并假设不同类别具有相近的协方差结构。这个假设越接近真实数据，线性判别轴越可靠。",
            },
            {
                "title": "投影公式",
                "body": "样本到判别轴的垂足由向量点积计算得到：先取样本相对轴上一点的向量，再投到单位方向向量上。这样得到的投影虚线彼此正交且平行。",
            },
            {
                "title": "阈值切分",
                "body": "投影完成后，模型会在一维轴上寻找能区分类别的阈值。二维空间里的最终决策边界，就是这个阈值在原空间中对应的线性切分。",
            },
            {
                "title": "观察重点",
                "body": "重点观察类别中心、投影轴方向、样本垂足和阈值位置。若投影后两类仍大量重叠，说明线性投影无法充分揭开类别差异。",
            },
        ],
        "deepDive": [
            {
                "title": "LDA 与 PCA 的区别",
                "body": "PCA 关心保留最大方差，不看标签；LDA 直接利用标签，寻找最有利于分类的方向。一个偏压缩，一个偏判别。",
            },
            {
                "title": "类间与类内散布",
                "body": "类间散布希望类别中心隔得远，类内散布希望同类点聚得紧。LDA 的目标就是让二者比例尽可能大。",
            },
            {
                "title": "正交投影的意义",
                "body": "样本被垂直投到判别轴上后，二维问题变成一维排序问题。投影点越分离，阈值越容易切开类别。",
            },
            {
                "title": "共享协方差假设",
                "body": "LDA 假设不同类别的散布形状相近。如果某类特别扁、某类特别散，线性判别轴可能会偏向错误方向。",
            },
            {
                "title": "结果怎么读",
                "body": "如果类别中心分开但召回率仍低，可能是类内散布太大；如果中心本身很近，说明当前特征对区分类别帮助有限。",
            },
        ],
    },
    "decision-tree": {
        "id": "decision-tree",
        "name": "决策树",
        "short_name": "Tree",
        "category": "树模型、集成学习与深度学习",
        "summary": "递归选择最优特征阈值，用水平或垂直切线划分特征空间。",
        "principle": [
            {
                "title": "核心思想",
                "body": "决策树把分类问题拆成一串 if-then 判断。每个节点选择一个特征和阈值，把样本切成更纯的两个子区域，直到满足停止条件或叶子足够纯。",
            },
            {
                "title": "切分准则",
                "body": "Gini、Entropy 和 Log Loss 都在衡量切分后的混杂程度。好的切分会让子节点里某一类占比更高，因此信息不确定性下降。",
            },
            {
                "title": "几何形态",
                "body": "普通决策树每次只看一个特征，所以二维空间里只能画水平线或垂直线。多层递归后，最终边界会形成一块块正交矩形区域。",
            },
            {
                "title": "复杂度控制",
                "body": "最大深度、叶子最小样本数会直接限制树的细碎程度。树越深越容易贴合训练集噪声，树太浅又可能无法表达真实结构。",
            },
            {
                "title": "观察重点",
                "body": "重点观察第一刀切在哪里、后续切线是否只在局部矩形内生效，以及最终边界是否过度碎片化。训练准确率高但测试表现差时，常见原因就是树太复杂。",
            },
        ],
        "deepDive": [
            {
                "title": "信息增益直觉",
                "body": "一次好的切分会让左右子节点更纯。纯度提升越明显，说明这个规则越能减少分类不确定性。",
            },
            {
                "title": "为什么边界是阶梯状",
                "body": "决策树每次只判断一个特征是否超过阈值，所以在二维空间里只能横切或竖切。多次局部切分叠起来就是阶梯状边界。",
            },
            {
                "title": "可解释性从哪里来",
                "body": "从根节点走到叶节点就是一条完整规则，例如“特征 A 小于某值，并且特征 B 大于某值，则预测为某类”。",
            },
            {
                "title": "过拟合风险",
                "body": "树越深，越容易为少数样本单独切出小区域。训练集表现可能很好，但测试集会因为规则过细而掉分。",
            },
            {
                "title": "结果怎么读",
                "body": "若训练准确率明显高于测试准确率，先看最大深度和叶子样本数；若两者都低，说明树太浅或特征本身区分度不足。",
            },
        ],
    },
    "adaboost": {
        "id": "adaboost",
        "name": "AdaBoost 集成学习",
        "short_name": "AdaBoost",
        "category": "树模型、集成学习与深度学习",
        "summary": "串行训练多个弱分类器，并在下一轮放大错分样本权重。",
        "principle": [
            {
                "title": "核心思想",
                "body": "AdaBoost 串行训练多个弱分类器。每一轮都会提高上一轮错分样本的权重，让后续分类器更关注困难样本，最后通过加权投票组成强分类器。",
            },
            {
                "title": "样本权重",
                "body": "样本权重表示当前模型的注意力。被错分的样本会在下一轮变大，说明它们正在影响下一条弱分类器切线的位置。",
            },
            {
                "title": "弱分类器权重",
                "body": "每个弱分类器自身也有权重。错误率低的弱分类器在最终投票中声音更大，错误率接近随机猜测的分类器贡献会变小。",
            },
            {
                "title": "边界组合",
                "body": "单个弱分类器通常只是简单横线或竖线，但多个弱分类器叠加后可以形成锯齿状组合边界，逐步修补上一轮留下的错误区域。",
            },
            {
                "title": "观察重点",
                "body": "重点观察错分样本何时变大、弱分类器是否围绕困难点调整，以及权重是否过度集中在少数异常点。噪声较多时，AdaBoost 可能被异常点牵着走。",
            },
        ],
        "deepDive": [
            {
                "title": "Boosting 的核心",
                "body": "Boosting 不是并行训练很多模型，而是一个接一个修正前面的错误。每一轮都带着上一轮的遗憾继续学习。",
            },
            {
                "title": "样本权重为何会变",
                "body": "错分样本权重变大，是为了让下一轮弱分类器更难忽略它们。权重变化就是模型注意力的迁移轨迹。",
            },
            {
                "title": "弱分类器也有权重",
                "body": "表现好的弱分类器在最终投票中权重大，表现差的权重小。强分类器不是简单平均，而是可信度加权组合。",
            },
            {
                "title": "对噪声敏感",
                "body": "如果某些样本本身标注错误或是异常点，AdaBoost 会反复关注它们，导致后续模型把精力浪费在噪声上。",
            },
            {
                "title": "结果怎么读",
                "body": "如果增加弱分类器后测试指标不再提升，说明模型已经进入收益平台期；若泛化差距扩大，就要降低学习率或减少轮数。",
            },
        ],
    },
    "mlp": {
        "id": "mlp",
        "name": "多层感知机",
        "short_name": "MLP",
        "category": "树模型、集成学习与深度学习",
        "summary": "通过多轮训练快照展示非线性边界逐渐形成的过程。",
        "principle": [
            {
                "title": "核心思想",
                "body": "MLP 由输入层、隐藏层和输出层组成。每层先做加权求和，再通过非线性激活函数变换，从而逐层构造更复杂的特征表达。",
            },
            {
                "title": "训练过程",
                "body": "模型先前向传播得到预测结果，再通过反向传播计算误差对每个权重的影响，最后由优化器更新参数。Epoch 快照展示的就是这个迭代过程。",
            },
            {
                "title": "非线性能力",
                "body": "隐藏层和激活函数让 MLP 不必局限于直线边界。它可以学习弯曲、包裹、分段平滑的决策区域，适合演示空间被逐步扭曲的过程。",
            },
            {
                "title": "超参数影响",
                "body": "隐藏层规模、激活函数、学习率、正则项和训练轮数都会影响边界形态。学习率过高可能震荡，训练过久且正则不足可能过拟合。",
            },
            {
                "title": "观察重点",
                "body": "重点观察早期边界是否接近线性、中期是否开始弯曲、最终是否平滑包裹异类样本。若边界出现很多无意义波纹，通常说明模型复杂度偏高。",
            },
        ],
        "deepDive": [
            {
                "title": "隐藏层在做什么",
                "body": "隐藏层可以看作把原始空间重新编码。经过多层非线性变换后，原本难以线性分开的样本可能在新表示中变得可分。",
            },
            {
                "title": "反向传播直觉",
                "body": "反向传播会计算每个权重对最终误差的责任大小。责任越大，更新越明显，模型就沿着降低损失的方向调整。",
            },
            {
                "title": "激活函数的作用",
                "body": "如果没有非线性激活，多层线性变换仍然等价于一层线性模型。激活函数是 MLP 能学习弯曲边界的关键。",
            },
            {
                "title": "训练不稳定的来源",
                "body": "学习率太大可能震荡，太小收敛慢；隐藏层过大可能过拟合，过小可能欠拟合。MLP 的表现高度依赖这些超参数。",
            },
            {
                "title": "结果怎么读",
                "body": "不要只看训练准确率。若训练集很高但测试集下降，说明边界可能出现无意义波纹；若两者都低，模型容量或训练轮数可能不足。",
            },
        ],
    },
}


@dataclass
class PreparedRun:
    dataset_id: str
    dataset: dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    labels: list[str]
    feature_names: list[str]
    feature_indices: list[int]
    bounds: dict[str, float]


def run_algorithm(
        algorithm_id: str,
        pattern: str = "blobs",
        hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hyperparameters = hyperparameters or {}
    data = _prepare_data(pattern)
    runners: dict[str, Callable[[PreparedRun, dict[str, Any]], list[dict[str, Any]]]] = {
        "knn": _run_knn,
        "logistic-regression": _run_logistic_regression,
        "lda": _run_lda,
        "decision-tree": _run_decision_tree,
        "adaboost": _run_adaboost,
        "mlp": _run_mlp,
    }
    steps = _renumber_steps(runners[algorithm_id](data, hyperparameters))
    algorithm = ALGORITHMS[algorithm_id]
    return {
        "algorithmId": algorithm_id,
        "algorithmName": algorithm["name"],
        "dataset": data.dataset,
        "labels": [
            {"index": index, "name": label, "color": PALETTE[index % len(PALETTE)]}
            for index, label in enumerate(data.labels)
        ],
        "coordinateSystem": {
            "space": "standardized-2d",
            "xFeatureIndex": data.feature_indices[0],
            "xFeatureName": data.feature_names[data.feature_indices[0]],
            "yFeatureIndex": data.feature_indices[1],
            "yFeatureName": data.feature_names[data.feature_indices[1]],
            "bounds": data.bounds,
        },
        "parameters": _public_parameters(algorithm_id, hyperparameters),
        "steps": steps,
        "analysis": _build_result_analysis(algorithm_id, data, steps),
    }


def _prepare_data(pattern: str) -> PreparedRun:
    """生成用于纯算法演示的 2D 合成二分类数据集"""
    n_samples = 150
    if pattern == "moons":
        x_raw, y = make_moons(n_samples=n_samples, noise=0.18, random_state=42)
        desc = "半月形非线性分布：专门测试算法是否具备解决非线性问题的能力（如 MLP、核技巧）。"
    elif pattern == "circles":
        x_raw, y = make_circles(n_samples=n_samples, noise=0.12, factor=0.4, random_state=42)
        desc = "环形包裹分布：中心被外环包裹，线性分类器在此分布下将完全失效。"
    else:
        # blobs (线性可分，增加少许重叠增加真实感)
        x_raw, y = make_blobs(n_samples=n_samples, centers=[[-2, -2], [2, 2]], cluster_std=1.5, random_state=42)
        desc = "高斯团簇分布：经典的近似线性可分场景，适合观察逻辑回归、LDA 等基础模型。"

    x = StandardScaler().fit_transform(x_raw)
    bounds = _bounds_for(x)

    return PreparedRun(
        dataset_id=pattern,
        dataset={"name": f"Synthetic 2D - {pattern.capitalize()}", "description": desc},
        x=x,
        y=y,
        labels=["类别 0 (Blue)", "类别 1 (Red)"],
        feature_names=["特征维度 X1", "特征维度 X2"],
        feature_indices=[0, 1],
        bounds=bounds,
    )


def _run_logistic_regression(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    learning_rate = _float_param(params, "learningRate", 0.12, 0.001, 2.0)
    iterations = _int_param(params, "iterations", 260, 30, 1000)
    l2 = _float_param(params, "l2", 0.001, 0.0, 2.0)
    seed = _int_param(params, "randomSeed", 7, 0, 100000)
    classes = len(data.labels)
    x_aug = np.column_stack([data.x, np.ones(len(data.x))])
    y_one_hot = np.eye(classes)[data.y]
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.35, size=(x_aug.shape[1], classes))

    snapshot_iterations = sorted(set([0, 10, 30, 80, 150, iterations]))
    snapshots = {}
    for iteration in range(iterations + 1):
        if iteration in snapshot_iterations:
            snapshots[iteration] = weights.copy()
        scores = x_aug @ weights
        probabilities = _softmax(scores)
        gradient = (x_aug.T @ (probabilities - y_one_hot)) / len(data.x)
        gradient[:2] += l2 * weights[:2]
        weights -= learning_rate * gradient

    steps = []
    for position, iteration in enumerate(snapshot_iterations):
        current = snapshots[iteration]
        predictions = np.argmax(x_aug @ current, axis=1)
        helpers = _logistic_boundary_helpers(current, data.bounds, data.labels)
        if position == 0:
            helpers.extend(_logistic_error_segments(data.x, data.y, current, data.labels, limit=70))
            description = "初始随机边界：后端随机生成 Softmax 权重，显示多分类线性边界和样本到相关边界的误差垂线。"
        elif iteration == iterations:
            description = "最终状态：梯度下降完成，后端用最终权重计算密集网格，形成概率热力图。"
        else:
            description = f"梯度下降第 {iteration} 次迭代：边界根据交叉熵梯度发生平移和旋转，继续寻找更低损失。"
        background = None
        if iteration == iterations:
            def proba_predictor(points, model=current):
                return _softmax(np.column_stack([points, np.ones(len(points))]) @ model)

            background = _grid(data.bounds, proba_predictor, data.labels, resolution=90, use_proba=True)

        steps.append(
            _step(
                description,
                _scatter(data, predictions=predictions),
                background,
                helpers,
                annotations=[{"text": f"iteration={iteration}", "tone": "neutral"}],
            )
        )
    return steps


def _run_lda(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    regularization = _float_param(params, "regularization", 0.001, 0.0, 1.0)
    labels = np.unique(data.y)
    overall_mean = np.mean(data.x, axis=0)
    centers = []
    sw = np.zeros((2, 2))
    sb = np.zeros((2, 2))
    for label in labels:
        points = data.x[data.y == label]
        center = np.mean(points, axis=0)
        centers.append(center)
        centered = points - center
        sw += centered.T @ centered
        delta = (center - overall_mean).reshape(2, 1)
        sb += len(points) * (delta @ delta.T)

    matrix = np.linalg.pinv(sw + np.eye(2) * regularization) @ sb
    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    axis = np.real(eigen_vectors[:, int(np.argmax(np.real(eigen_values)))])
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    if axis[0] < 0:
        axis = -axis

    center_scalars = np.asarray([float((center - overall_mean) @ axis) for center in centers])
    ordered = list(np.argsort(center_scalars))
    thresholds = [
        float((center_scalars[ordered[index]] + center_scalars[ordered[index + 1]]) / 2)
        for index in range(len(ordered) - 1)
    ]

    def predict(points: np.ndarray) -> np.ndarray:
        scalars = (points - overall_mean) @ axis
        distances = np.abs(scalars[:, None] - center_scalars[None, :])
        return np.argmin(distances, axis=1)

    center_helpers = [
        {
            "type": "point",
            "x": _num(center[0]),
            "y": _num(center[1]),
            "radius": 12,
            "color": PALETTE[int(label) % len(PALETTE)],
            "label": f"{data.labels[int(label)]} 中心",
        }
        for label, center in zip(labels, centers)
    ]
    axis_line = _line_from_point_direction(overall_mean, axis, data.bounds)
    projection_helpers = _lda_projection_helpers(data.x, overall_mean, axis, data.y, data.labels)
    threshold_helpers = _lda_threshold_helpers(overall_mean, axis, thresholds, data.bounds)

    return [
        _step("计算类别中心点：计算二分类的各自几何中心坐标。", _scatter(data), None, center_helpers),
        _step("最优投影轴：寻找让类间距离最大、类内方差最小的一维判别轴。", _scatter(data), None,
              center_helpers + [axis_line]),
        _step("正交投影：计算每个样本到判别轴的垂足。", _scatter(data), None,
              center_helpers + [axis_line] + projection_helpers),
        _step(
            "轴上阈值切分：在投影轴上取阈值形成线性分类面。",
            _scatter(data, predictions=predict(data.x)),
            _grid(data.bounds, predict, data.labels, resolution=90),
            center_helpers + [axis_line] + threshold_helpers,
        ),
    ]


def _run_decision_tree(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    criterion = str(params.get("criterion", "gini"))
    if criterion not in {"gini", "entropy", "log_loss"}:
        criterion = "gini"
    max_depth = _int_param(params, "maxDepth", 4, 1, 8)
    min_samples_leaf = _int_param(params, "minSamplesLeaf", 3, 1, 30)
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                   random_state=42)
    model.fit(data.x, data.y)
    splits = []
    _collect_tree_splits(model, 0, data.bounds.copy(), splits, data.labels)
    if not splits:
        return [_step("当前限制下未能形成切分。", _scatter(data, predictions=model.predict(data.x)),
                      _grid(data.bounds, model.predict, data.labels, resolution=90), [])]

    steps = []
    visible_splits = splits[: min(10, len(splits))]
    for index, split in enumerate(visible_splits):
        helpers = [_tree_split_helper(item, data.bounds) for item in visible_splits[: index + 1]]
        steps.append(
            _step(
                f"第 {index + 1} 次递归切分：寻找信息增益最大的阈值加入一条正交切线。",
                _scatter(data),
                None,
                helpers,
                annotations=[{"text": f"{split['featureName']} <= {_num(split['threshold'])}", "tone": "neutral"}],
            )
        )

    steps.append(
        _step(
            "决策树空间：形成典型的正交矩形决策边界。",
            _scatter(data, predictions=model.predict(data.x)),
            _grid(data.bounds, model.predict, data.labels, resolution=90),
            [_tree_split_helper(item, data.bounds) for item in splits],
        )
    )
    return steps


def _run_knn(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    k = _int_param(params, "k", 5, 1, min(35, len(data.x) - 1))
    metric = str(params.get("distanceMetric", "euclidean"))
    if metric not in {"euclidean", "manhattan"}:
        metric = "euclidean"
    unknown_index = int(np.argmin(np.linalg.norm(data.x - np.mean(data.x, axis=0), axis=1)))
    mask = np.ones(len(data.x), dtype=bool)
    mask[unknown_index] = False
    train_x = data.x[mask]
    train_y = data.y[mask]
    unknown = data.x[unknown_index]
    if metric == "manhattan":
        distances = np.sum(np.abs(train_x - unknown), axis=1)
    else:
        distances = np.linalg.norm(train_x - unknown, axis=1)
    order = np.argsort(distances)
    selected = order[:k]
    votes = []
    for label in range(len(data.labels)):
        votes.append(
            {"labelIndex": label, "labelName": data.labels[label], "votes": int(np.sum(train_y[selected] == label))})
    votes = sorted(votes, key=lambda item: (-item["votes"], item["labelIndex"]))
    winner = votes[0]
    model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    model.fit(train_x, train_y)

    known_points = _scatter_from_arrays(train_x, train_y, data.labels)
    unknown_point = _unknown_point(unknown, "unknown-0")
    all_lines = [
        {
            "type": "segment", "x1": _num(unknown[0]), "y1": _num(unknown[1]),
            "x2": _num(train_x[index][0]), "y2": _num(train_x[index][1]),
            "color": "rgba(100,116,139,0.28)", "width": 1, "dash": [5, 7]
        } for index in order
    ]
    selected_lines = [
        {
            "type": "segment", "x1": _num(unknown[0]), "y1": _num(unknown[1]),
            "x2": _num(train_x[index][0]), "y2": _num(train_x[index][1]),
            "color": PALETTE[int(train_y[index]) % len(PALETTE)], "width": 2.6
        } for index in selected
    ]
    selected_ids = {f"sample-{int(index)}" for index in selected}
    selected_scatter = [{**pt, "selected": True, "radius": 10} if pt["id"] in selected_ids else pt for pt in
                        known_points]

    kth_distance = float(distances[order[k - 1]])
    circle_helper = {
        "type": "circle", "x": _num(unknown[0]), "y": _num(unknown[1]),
        "r": _num(kth_distance), "color": "#94a3b8", "dash": [4, 4], "fill": "rgba(148, 163, 184, 0.15)"
    }

    return [
        _step("距离度量：计算未知点到所有已知样本的距离。", known_points + [unknown_point], None, all_lines),
        _step(
            f"选择最近的 K={k} 个邻居：以第 K 个样本距离为半径画圆并投票。",
            selected_scatter + [{**unknown_point, "predictedLabelIndex": winner["labelIndex"],
                                 "predictedLabelName": winner["labelName"]}],
            None, selected_lines + [circle_helper]
        ),
        _step(
            "KNN 决策背景：逐点计算最近邻展示几何边界形态。",
            selected_scatter + [{**unknown_point, "predictedLabelIndex": winner["labelIndex"]}],
            _grid(data.bounds, model.predict, data.labels, resolution=90),
            selected_lines + [circle_helper]
        ),
    ]


def _run_adaboost(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = _int_param(params, "nEstimators", 5, 1, 12)
    learning_rate = _float_param(params, "learningRate", 0.8, 0.05, 3.0)
    classes = len(data.labels)
    weights = np.ones(len(data.x)) / len(data.x)
    estimators, alphas, steps = [], [], []
    steps.append(_step("初始权重：所有样本权重一致。", _scatter(data, weights=weights), None, []))

    for round_index in range(rounds):
        stump = DecisionTreeClassifier(max_depth=1, random_state=100 + round_index)
        stump.fit(data.x, data.y, sample_weight=weights)
        predictions = stump.predict(data.x)
        incorrect = predictions != data.y
        error = max(min(float(np.sum(weights[incorrect])), 1 - 1e-9), 1e-9)
        alpha = learning_rate * max(0.35, math.log((1 - error) / error) + math.log(max(classes - 1, 1)))

        helper = _stump_helper(stump, data.bounds)
        steps.append(_step(f"第 {round_index + 1} 个弱分类器切分",
                           _scatter(data, weights=weights, predictions=predictions, misclassified=incorrect), None,
                           [helper]))

        weights = weights * np.exp(alpha * incorrect)
        weights /= np.sum(weights)
        estimators.append(stump)
        alphas.append(alpha)
        steps.append(_step(f"第 {round_index + 1} 轮权重放大：错分样本散点变大，下一轮优先关注。",
                           _scatter(data, weights=weights, predictions=predictions, misclassified=incorrect), None,
                           [helper]))

    def strong_predict(points: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(points), classes))
        for stump, alpha in zip(estimators, alphas):
            scores[np.arange(len(points)), stump.predict(points)] += alpha
        return np.argmax(scores, axis=1)

    steps.append(
        _step(
            "强分类器组合：弱分类器加权投票形成锯齿状最终边界。",
            _scatter(data, weights=weights, predictions=strong_predict(data.x)),
            _grid(data.bounds, strong_predict, data.labels, resolution=90),
            [_stump_helper(stump, data.bounds, color="#0f172a", width=1.6) for stump in estimators],
        )
    )
    return steps


def _run_mlp(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    epochs = _int_param(params, "epochs", 140, 10, 400)
    hidden_layers = _hidden_layers(params.get("hiddenLayers", "32,16"))
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers, activation=params.get("activation", "tanh"),
        solver="adam", alpha=_float_param(params, "alpha", 0.0003, 0.0, 1.0),
        learning_rate_init=_float_param(params, "learningRate", 0.025, 0.0001, 1.0),
        max_iter=1, random_state=_int_param(params, "randomSeed", 42, 0, 100000),
    )
    snapshot_epochs = sorted(set([1, 3, 8, 20, 45, 90, epochs]))
    steps = []
    classes = np.arange(len(data.labels))
    for epoch in range(1, epochs + 1):
        model.partial_fit(data.x, data.y, classes=classes) if epoch == 1 else model.partial_fit(data.x, data.y)
        if epoch in snapshot_epochs:
            steps.append(
                _step(
                    f"Epoch {epoch} 快照：显示网络逐渐扭曲空间形成的非线性概率热力图。",
                    _scatter(data, predictions=model.predict(data.x)),
                    _grid(data.bounds, model.predict_proba, data.labels, resolution=90, use_proba=True),
                    [],
                )
            )
    return steps


# ---------------- 辅助方法与底层计算 ----------------

def _public_parameters(algorithm_id: str, params: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "knn": {"k": 5, "distanceMetric": "euclidean"},
        "logistic-regression": {"learningRate": 0.12, "iterations": 260, "l2": 0.001, "randomSeed": 7},
        "lda": {"regularization": 0.001, "thresholdMode": "nearest-center"},
        "decision-tree": {"criterion": "gini", "maxDepth": 4, "minSamplesLeaf": 3},
        "adaboost": {"nEstimators": 5, "learningRate": 0.8},
        "mlp": {"hiddenLayers": "32,16", "activation": "tanh", "learningRate": 0.025, "alpha": 0.0003, "epochs": 140,
                "randomSeed": 42},
    }
    merged = defaults[algorithm_id].copy()
    merged.update(params)
    return merged


def _build_result_analysis(algorithm_id: str, data: PreparedRun, steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"title": "独立演示模块声明",
         "body": "当前展示已脱离真实业务数据，纯粹使用 Scikit-learn 的二维二分类合成数据，专门用于暴露算法的几何几何切分本质。"},
        {"title": "结果限定",
         "body": "由于降维过程在实际中会导致巨大信息丢失，因此本页面使用合成的 2D 坐标（特征 X1 与 X2）来确保绘图能完美呈现出纯正的数学决策边界形态。"},
    ]


def _step(desc: str, scatter: list, bg: dict | None, helpers: list, annotations: list = None) -> dict:
    return {"index": 0, "description": desc, "scatter": scatter, "backgroundGrid": bg, "helpers": helpers,
            "annotations": annotations or []}


def _renumber_steps(steps: list) -> list:
    for i, s in enumerate(steps): s["index"] = i
    return steps


def _bounds_for(points: np.ndarray) -> dict:
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    pad_x, pad_y = max((x_max - x_min) * 0.18, 0.8), max((y_max - y_min) * 0.18, 0.8)
    return {"xMin": _num(x_min - pad_x), "xMax": _num(x_max + pad_x), "yMin": _num(y_min - pad_y),
            "yMax": _num(y_max + pad_y)}


def _scatter(data, weights=None, predictions=None, misclassified=None) -> list:
    return _scatter_from_arrays(data.x, data.y, data.labels, weights, predictions, misclassified)


def _scatter_from_arrays(x, y, labels, weights=None, predictions=None, misclassified=None) -> list:
    if weights is None: weights = np.ones(len(x)) / max(len(x), 1)
    relative = weights / max(float(np.mean(weights)), 1e-12)
    pts = []
    for i, (pt, label) in enumerate(zip(x, y)):
        item = {
            "id": f"sample-{i}", "x": _num(pt[0]), "y": _num(pt[1]), "labelIndex": int(label),
            "color": PALETTE[int(label) % len(PALETTE)],
            "radius": _num(5.8 + min(13.0, max(0.0, math.sqrt(float(relative[i])) - 1.0) * 5.2)),
            "selected": False, "misclassified": bool(misclassified[i]) if misclassified is not None else False
        }
        if predictions is not None:
            item.update({"predictedColor": PALETTE[int(predictions[i]) % len(PALETTE)]})
        pts.append(item)
    return pts


def _unknown_point(point: np.ndarray, item_id: str) -> dict:
    return {"id": item_id, "x": _num(point[0]), "y": _num(point[1]), "labelIndex": -1, "color": "#111827", "radius": 11,
            "selected": True}


def _grid(bounds: dict, predictor: Callable, labels: list, resolution: int = 90, use_proba: bool = False) -> dict:
    x_values = np.linspace(bounds["xMin"], bounds["xMax"], resolution)
    y_values = np.linspace(bounds["yMin"], bounds["yMax"], resolution)
    xx, yy = np.meshgrid(x_values, y_values)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    if use_proba:
        probas = predictor(flat)
        preds = np.argmax(probas, axis=1)
        max_p = np.max(probas, axis=1)
        alphas = np.clip((max_p - 0.5) / 0.5, 0.0, 1.0)
    else:
        preds = np.asarray(predictor(flat), dtype=int)
        alphas = [None] * len(preds)
    return {
        "columns": resolution, "rows": resolution,
        "xValues": [_num(v) for v in x_values], "yValues": [_num(v) for v in y_values],
        "points": [
            {
                "x": _num(pt[0]), "y": _num(pt[1]), "color": PALETTE[int(label) % len(PALETTE)],
                **({"alpha": _num(float(a))} if a is not None else {})
            }
            for pt, label, a in zip(flat, preds, alphas)
        ],
    }


def _softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _logistic_boundary_helpers(weights, bounds, labels) -> list:
    return [_line_from_coefficients(float(weights[0, 0] - weights[0, 1]), float(weights[1, 0] - weights[1, 1]),
                                    float(weights[2, 0] - weights[2, 1]), bounds) or {}]


def _logistic_error_segments(x, y, weights, labels, limit) -> list:
    return []  # 为保持清爽，已在二分类独立演示中省略该垂线计算


def _lda_projection_helpers(x, origin, axis, y, labels) -> list:
    helpers = []
    for i, pt in enumerate(x):
        proj = origin + float((pt - origin) @ axis) * axis
        helpers.extend([
            {"type": "segment", "x1": _num(pt[0]), "y1": _num(pt[1]), "x2": _num(proj[0]), "y2": _num(proj[1]),
             "color": "rgba(15,23,42,0.30)", "width": 1, "dash": [3, 5]},
            {"type": "point", "x": _num(proj[0]), "y": _num(proj[1]), "radius": 3.6,
             "color": PALETTE[int(y[i]) % len(PALETTE)]}
        ])
    return helpers


def _lda_threshold_helpers(origin, axis, thresholds, bounds) -> list:
    helpers = []
    normal = np.array([-axis[1], axis[0]])
    for th in thresholds:
        pt = origin + th * axis
        line = _line_from_point_direction(pt, normal, bounds)
        line.update({"color": "#0f172a", "width": 1.7, "dash": [8, 5]})
        helpers.extend([line, {"type": "point", "x": _num(pt[0]), "y": _num(pt[1]), "radius": 6, "color": "#0f172a"}])
    return helpers


def _collect_tree_splits(model, node_id, region, splits, labels) -> None:
    tree = model.tree_
    if tree.children_left[node_id] == tree.children_right[node_id]: return
    feature, threshold = int(tree.feature[node_id]), float(tree.threshold[node_id])
    splits.append({"feature": feature, "featureName": "X1" if feature == 0 else "X2", "threshold": threshold,
                   "region": region.copy()})
    left_region, right_region = region.copy(), region.copy()
    if feature == 0:
        left_region["xMax"], right_region["xMin"] = min(left_region["xMax"], threshold), max(right_region["xMin"],
                                                                                             threshold)
    else:
        left_region["yMax"], right_region["yMin"] = min(left_region["yMax"], threshold), max(right_region["yMin"],
                                                                                             threshold)
    _collect_tree_splits(model, tree.children_left[node_id], left_region, splits, labels)
    _collect_tree_splits(model, tree.children_right[node_id], right_region, splits, labels)


def _tree_split_helper(split, bounds) -> dict:
    r = split["region"]
    if split["feature"] == 0:
        return {"type": "segment", "x1": _num(split["threshold"]), "y1": _num(r["yMin"]),
                "x2": _num(split["threshold"]), "y2": _num(r["yMax"]), "color": "#0f172a", "width": 2.4}
    return {"type": "segment", "x1": _num(r["xMin"]), "y1": _num(split["threshold"]), "x2": _num(r["xMax"]),
            "y2": _num(split["threshold"]), "color": "#0f172a", "width": 2.4}


def _stump_helper(stump, bounds, color="#dc2626", width=2.5) -> dict:
    f, t = int(stump.tree_.feature[0]), float(stump.tree_.threshold[0])
    if f == 0: return {"type": "segment", "x1": _num(t), "y1": bounds["yMin"], "x2": _num(t), "y2": bounds["yMax"],
                       "color": color, "width": width}
    return {"type": "segment", "x1": bounds["xMin"], "y1": _num(t), "x2": bounds["xMax"], "y2": _num(t), "color": color,
            "width": width}


def _line_from_point_direction(pt, dir, bounds) -> dict:
    return _line_from_coefficients(float(-dir[1]), float(dir[0]), float(dir[1] * pt[0] - dir[0] * pt[1]), bounds) or {
        "type": "segment", "x1": pt[0] - dir[0], "y1": pt[1] - dir[1], "x2": pt[0] + dir[0], "y2": pt[1] + dir[1]}


def _line_from_coefficients(a, b, c, bounds) -> dict | None:
    pts = []
    for x in (bounds["xMin"], bounds["xMax"]):
        if abs(b) > 1e-12 and bounds["yMin"] - 1e-9 <= (-a * x - c) / b <= bounds["yMax"] + 1e-9: pts.append(
            (x, (-a * x - c) / b))
    for y in (bounds["yMin"], bounds["yMax"]):
        if abs(a) > 1e-12 and bounds["xMin"] - 1e-9 <= (-b * y - c) / a <= bounds["xMax"] + 1e-9: pts.append(
            ((-b * y - c) / a, y))
    unique = list(set([(_num(p[0]), _num(p[1])) for p in pts]))
    return {"type": "segment", "x1": unique[0][0], "y1": unique[0][1], "x2": unique[1][0], "y2": unique[1][1],
            "color": "#0f172a", "width": 2.4} if len(unique) >= 2 else None


def _hidden_layers(val) -> tuple:
    return tuple([min(max(int(float(i)), 2), 128) for i in str(val).replace("，", ",").split(",") if i.strip()][:3]) or (
        32, 16)


def _int_param(p, k, d, m, x): return min(
    max(int(p.get(k, d) if str(p.get(k, d)).replace('-', '').isdigit() else d), m), x)


def _float_param(p, k, d, m, x): return min(
    max(float(p.get(k, d) if str(p.get(k, d)).replace('.', '', 1).replace('-', '').isdigit() else d), m), x)


def _num(v): return round(float(v), 5) if math.isfinite(float(v)) else 0.0
