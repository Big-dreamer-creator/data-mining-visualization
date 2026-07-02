from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core.datasets import get_dataset_summary, load_dataset


PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
DEFAULT_FEATURES = {
    "iris": [2, 3],
    "wine": [6, 12],
}


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
                "body": "KNN 不学习参数，而是在预测时计算未知点与所有已知样本的距离，再由最近的 K 个样本投票。",
            },
            {
                "title": "过程观察",
                "body": "可视化重点是距离连线、最近邻集合和投票结果如何一步步产生，而不是只看最终类别。",
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
                "body": "逻辑回归用线性得分函数表示类别边界，通过最小化交叉熵不断移动和旋转边界。",
            },
            {
                "title": "过程观察",
                "body": "可视化重点是初始随机边界、梯度下降过程中的边界变化，以及样本到边界的误差垂线。",
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
                "body": "LDA 通过类别中心和类内散布矩阵求出最优判别方向，让类别投影后尽可能分开。",
            },
            {
                "title": "过程观察",
                "body": "可视化重点是类别中心、最优投影轴、正交投影点和轴上阈值如何形成最终线性决策。",
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
                "body": "决策树每次选择能让子区域类别更纯的切分规则，递归形成一组轴向矩形区域。",
            },
            {
                "title": "过程观察",
                "body": "可视化重点是每一刀切在哪里、属于哪个矩形区域，以及最终边界为何呈俄罗斯方块状。",
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
                "body": "AdaBoost 会提高上一轮错分样本的权重，使后续弱分类器更关注困难样本。",
            },
            {
                "title": "过程观察",
                "body": "可视化重点是弱分类器切线、错分点变大和多条切线组合成强分类器边界的过程。",
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
                "body": "MLP 通过隐藏层和非线性激活函数学习弯曲的分类边界，并用反向传播逐步更新权重。",
            },
            {
                "title": "过程观察",
                "body": "可视化重点是不同 epoch 的边界快照，观察空间如何从僵硬分割逐渐变得平滑非线性。",
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
    dataset_id: str,
    feature_indices: list[int] | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hyperparameters = hyperparameters or {}
    data = _prepare_data(dataset_id, feature_indices)
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


def _prepare_data(dataset_id: str, feature_indices: list[int] | None) -> PreparedRun:
    loaded = load_dataset(dataset_id)
    feature_names = loaded["featureNames"]
    indices = feature_indices or DEFAULT_FEATURES.get(dataset_id, [0, 1])
    if len(indices) != 2:
        raise ValueError("featureIndices must contain exactly two feature indexes")
    if indices[0] == indices[1]:
        raise ValueError("featureIndices must reference two different features")
    if min(indices) < 0 or max(indices) >= len(feature_names):
        raise ValueError("featureIndices contains an out-of-range feature index")

    x_raw = loaded["data"][:, indices]
    x = StandardScaler().fit_transform(x_raw)
    y = np.asarray(loaded["target"], dtype=int)
    bounds = _bounds_for(x)
    return PreparedRun(
        dataset_id=dataset_id,
        dataset=get_dataset_summary(dataset_id, include_preview=True),
        x=x,
        y=y,
        labels=loaded["targetNames"],
        feature_names=feature_names,
        feature_indices=indices,
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
            description = "最终状态：梯度下降完成，后端用最终权重计算密集网格，形成线性多边形决策区域。"
        else:
            description = f"梯度下降第 {iteration} 次迭代：边界根据交叉熵梯度发生平移和旋转，继续寻找更低损失。"
        background = None
        if iteration == iterations:
            background = _grid(
                data.bounds,
                lambda points, model=current: np.argmax(
                    np.column_stack([points, np.ones(len(points))]) @ model,
                    axis=1,
                ),
                data.labels,
                resolution=68,
            )
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
        _step(
            "计算类别中心点：后端在二维标准化特征空间中求出每个类别的真实中心坐标。",
            _scatter(data),
            None,
            center_helpers,
        ),
        _step(
            "计算一维最优投影轴：后端求解类间散布与类内散布的广义方向，输出判别轴绝对坐标。",
            _scatter(data),
            None,
            center_helpers + [axis_line],
        ),
        _step(
            "正交投影：后端使用向量点积公式计算每个样本到判别轴的真实垂足坐标，虚线因此完全平行。",
            _scatter(data),
            None,
            center_helpers + [axis_line] + projection_helpers,
        ),
        _step(
            "轴上阈值切分：后端在投影轴上按类别中心顺序取阈值，并生成最终线性决策背景网格。",
            _scatter(data, predictions=predict(data.x)),
            _grid(data.bounds, predict, data.labels, resolution=68),
            center_helpers + [axis_line] + threshold_helpers,
            annotations=[
                {"text": f"阈值数量：{len(thresholds)}", "tone": "neutral"},
                {"text": "预测规则：样本投影后归入最近的类别中心区间。", "tone": "neutral"},
            ],
        ),
    ]


def _run_decision_tree(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    criterion = str(params.get("criterion", "gini"))
    if criterion not in {"gini", "entropy", "log_loss"}:
        criterion = "gini"
    max_depth = _int_param(params, "maxDepth", 4, 1, 8)
    min_samples_leaf = _int_param(params, "minSamplesLeaf", 3, 1, 30)
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(data.x, data.y)
    splits = []
    _collect_tree_splits(model, 0, data.bounds.copy(), splits, data.labels)
    if not splits:
        predictions = model.predict(data.x)
        return [
            _step(
                "决策树没有找到有效切分：当前特征组合下根节点已经形成叶节点。",
                _scatter(data, predictions=predictions),
                _grid(data.bounds, model.predict, data.labels, resolution=68),
                [],
            )
        ]

    steps = []
    visible_splits = splits[: min(10, len(splits))]
    for index, split in enumerate(visible_splits):
        helpers = [_tree_split_helper(item, data.bounds) for item in visible_splits[: index + 1]]
        prefix = "第一刀" if index == 0 else f"第 {index + 1} 次递归切分"
        axis_name = "垂直" if split["feature"] == 0 else "水平"
        steps.append(
            _step(
                f"{prefix}：后端选择当前节点信息增益最大的特征阈值，在已有矩形区域内加入一条{axis_name}切线。",
                _scatter(data),
                None,
                helpers,
                annotations=[
                    {
                        "text": f"{split['featureName']} <= {_num(split['threshold'])}",
                        "tone": "neutral",
                    }
                ],
            )
        )

    predictions = model.predict(data.x)
    steps.append(
        _step(
            "最终决策树空间：所有递归切分完成，后端网格预测呈现典型的正交矩形决策边界。",
            _scatter(data, predictions=predictions),
            _grid(data.bounds, model.predict, data.labels, resolution=74),
            [_tree_split_helper(item, data.bounds) for item in splits],
            annotations=[
                {"text": f"树深度：{model.get_depth()}，叶节点：{model.get_n_leaves()}", "tone": "neutral"}
            ],
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
        votes.append({"labelIndex": label, "labelName": data.labels[label], "votes": int(np.sum(train_y[selected] == label))})
    votes = sorted(votes, key=lambda item: (-item["votes"], item["labelIndex"]))
    winner = votes[0]
    model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    model.fit(train_x, train_y)

    known_points = _scatter_from_arrays(train_x, train_y, data.labels)
    unknown_point = _unknown_point(unknown, "unknown-0")
    all_lines = [
        {
            "type": "segment",
            "x1": _num(unknown[0]),
            "y1": _num(unknown[1]),
            "x2": _num(train_x[index][0]),
            "y2": _num(train_x[index][1]),
            "color": "rgba(100,116,139,0.28)",
            "width": 1,
            "dash": [5, 7],
            "label": f"distance={_num(distances[index])}",
            "distance": _num(distances[index]),
        }
        for index in order
    ]
    selected_lines = []
    for rank, index in enumerate(selected):
        selected_lines.append(
            {
                "type": "segment",
                "x1": _num(unknown[0]),
                "y1": _num(unknown[1]),
                "x2": _num(train_x[index][0]),
                "y2": _num(train_x[index][1]),
                "color": PALETTE[int(train_y[index]) % len(PALETTE)],
                "width": 2.6,
                "rank": rank + 1,
                "distance": _num(distances[index]),
                "label": f"#{rank + 1} {data.labels[int(train_y[index])]}",
            }
        )
    selected_ids = {f"sample-{int(index)}" for index in selected}
    selected_scatter = []
    for point_index, point in enumerate(known_points):
        if point["id"] in selected_ids:
            point = {**point, "selected": True, "radius": 10}
        selected_scatter.append(point)

    return [
        _step(
            "距离度量：后端预设一个未知样本点，并计算它到所有已知训练样本的真实距离连线。",
            known_points + [unknown_point],
            None,
            all_lines,
            annotations=[{"text": f"未知样本真实类别在预测前隐藏：{data.labels[int(data.y[unknown_index])]}", "tone": "muted"}],
        ),
        _step(
            f"选择最近的 K={k} 个邻居：后端按距离排序并返回最近邻集合，同时完成类别投票统计。",
            selected_scatter + [{**unknown_point, "predictedLabelIndex": winner["labelIndex"], "predictedLabelName": winner["labelName"]}],
            None,
            selected_lines,
            annotations=[
                {"text": "投票结果：" + " / ".join(f"{item['labelName']} {item['votes']}票" for item in votes), "tone": "neutral"},
                {"text": f"预测类别：{winner['labelName']}", "tone": "success"},
            ],
        ),
        _step(
            "KNN 决策背景：后端对密集网格逐点计算最近邻投票，展示 K 值带来的破碎或平滑边界。",
            selected_scatter + [{**unknown_point, "predictedLabelIndex": winner["labelIndex"], "predictedLabelName": winner["labelName"]}],
            _grid(data.bounds, model.predict, data.labels, resolution=72),
            selected_lines,
            annotations=[{"text": f"距离度量：{metric}", "tone": "neutral"}],
        ),
    ]


def _run_adaboost(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = _int_param(params, "nEstimators", 5, 1, 12)
    learning_rate = _float_param(params, "learningRate", 0.8, 0.05, 3.0)
    classes = len(data.labels)
    weights = np.ones(len(data.x)) / len(data.x)
    estimators = []
    alphas = []
    steps = [
        _step(
            "初始权重：每个样本权重完全一致，散点半径相同，第一轮弱分类器尚未开始关注困难样本。",
            _scatter(data, weights=weights),
            None,
            [],
            annotations=[{"text": "所有样本权重 = 1 / 样本数", "tone": "neutral"}],
        )
    ]

    for round_index in range(rounds):
        stump = DecisionTreeClassifier(max_depth=1, random_state=100 + round_index)
        stump.fit(data.x, data.y, sample_weight=weights)
        predictions = stump.predict(data.x)
        incorrect = predictions != data.y
        error = float(np.sum(weights[incorrect]))
        error = min(max(error, 1e-9), 1 - 1e-9)
        raw_alpha = math.log((1 - error) / error) + math.log(max(classes - 1, 1))
        alpha = learning_rate * max(0.35, raw_alpha)
        helper = _stump_helper(stump, data.bounds)
        steps.append(
            _step(
                f"第 {round_index + 1} 个弱分类器：后端训练一条简单水平或垂直切线，并标出本轮错分样本。",
                _scatter(data, weights=weights, predictions=predictions, misclassified=incorrect),
                None,
                [helper],
                annotations=[
                    {"text": f"加权错误率={_num(error)}，分类器权重 alpha={_num(alpha)}", "tone": "neutral"}
                ],
            )
        )
        weights = weights * np.exp(alpha * incorrect)
        weights = weights / np.sum(weights)
        estimators.append(stump)
        alphas.append(alpha)
        steps.append(
            _step(
                f"第 {round_index + 1} 轮权重更新：上一帧被错分的样本权重显著变大，下一轮会优先关注这些点。",
                _scatter(data, weights=weights, predictions=predictions, misclassified=incorrect),
                None,
                [helper],
                annotations=[
                    {"text": f"最大权重 / 平均权重 = {_num(np.max(weights) / np.mean(weights))}", "tone": "warning"}
                ],
            )
        )

    def strong_predict(points: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(points), classes))
        for stump, alpha in zip(estimators, alphas):
            pred = stump.predict(points)
            scores[np.arange(len(points)), pred] += alpha
        return np.argmax(scores, axis=1)

    steps.append(
        _step(
            "强分类器组合边界：后端将所有弱分类器加权投票，对网格逐点预测后形成锯齿状组合边界。",
            _scatter(data, weights=weights, predictions=strong_predict(data.x)),
            _grid(data.bounds, strong_predict, data.labels, resolution=74),
            [_stump_helper(stump, data.bounds, color="#0f172a", width=1.6) for stump in estimators],
            annotations=[{"text": f"弱分类器数量：{len(estimators)}", "tone": "neutral"}],
        )
    )
    return steps


def _run_mlp(data: PreparedRun, params: dict[str, Any]) -> list[dict[str, Any]]:
    epochs = _int_param(params, "epochs", 140, 10, 400)
    hidden_layers = _hidden_layers(params.get("hiddenLayers", "32,16"))
    activation = str(params.get("activation", "tanh"))
    if activation not in {"identity", "logistic", "tanh", "relu"}:
        activation = "tanh"
    learning_rate = _float_param(params, "learningRate", 0.025, 0.0001, 1.0)
    alpha = _float_param(params, "alpha", 0.0003, 0.0, 1.0)
    seed = _int_param(params, "randomSeed", 42, 0, 100000)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate,
        max_iter=1,
        random_state=seed,
    )
    snapshot_epochs = sorted(set([1, 3, 8, 20, 45, 90, epochs]))
    steps = []
    classes = np.arange(len(data.labels))
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            model.partial_fit(data.x, data.y, classes=classes)
        else:
            model.partial_fit(data.x, data.y)
        if epoch in snapshot_epochs:
            predictions = model.predict(data.x)
            boundary_note = "早期边界仍较僵硬，通常接近线性或大块区域。" if epoch <= 8 else "边界开始被隐藏层逐步扭曲。" if epoch < epochs else "最终快照呈现平滑非线性决策区域。"
            steps.append(
                _step(
                    f"Epoch {epoch} 快照：后端保存当前神经网络参数，并对密集网格执行预测。{boundary_note}",
                    _scatter(data, predictions=predictions),
                    _grid(data.bounds, model.predict, data.labels, resolution=70),
                    [],
                    annotations=[
                        {"text": f"loss={_num(float(model.loss_))}", "tone": "neutral"},
                        {"text": f"隐藏层={hidden_layers}", "tone": "neutral"},
                    ],
                )
            )
    return steps


def _public_parameters(algorithm_id: str, params: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "knn": {"k": 5, "distanceMetric": "euclidean"},
        "logistic-regression": {"learningRate": 0.12, "iterations": 260, "l2": 0.001, "randomSeed": 7},
        "lda": {"regularization": 0.001, "thresholdMode": "nearest-center"},
        "decision-tree": {"criterion": "gini", "maxDepth": 4, "minSamplesLeaf": 3},
        "adaboost": {"nEstimators": 5, "learningRate": 0.8},
        "mlp": {"hiddenLayers": "32,16", "activation": "tanh", "learningRate": 0.025, "alpha": 0.0003, "epochs": 140, "randomSeed": 42},
    }
    merged = defaults[algorithm_id].copy()
    merged.update(params)
    return merged


def _build_result_analysis(
    algorithm_id: str,
    data: PreparedRun,
    steps: list[dict[str, Any]],
) -> list[dict[str, str]]:
    final_step = steps[-1] if steps else _step("", [], None, [])
    predicted_points = [
        point
        for point in final_step.get("scatter", [])
        if point.get("labelIndex", -1) >= 0 and "predictedLabelIndex" in point
    ]
    if predicted_points:
        correct = sum(
            1 for point in predicted_points
            if int(point["labelIndex"]) == int(point["predictedLabelIndex"])
        )
        accuracy = correct / max(len(predicted_points), 1)
        result_body = (
            f"最终帧在当前二维特征空间内对 {len(predicted_points)} 个样本给出预测，"
            f"其中 {correct} 个与真实类别一致，帧内准确率为 {accuracy:.2%}。"
        )
    else:
        result_body = (
            "最终帧重点展示单个未知样本的预测过程。该算法页面的结果解读应结合最近邻投票、"
            "邻居类别分布和网格背景，而不是只看一个测试点是否分类正确。"
        )

    grid_points = final_step.get("backgroundGrid", {}).get("points", []) if final_step.get("backgroundGrid") else []
    if grid_points:
        counts = Counter(point["labelName"] for point in grid_points)
        total = sum(counts.values())
        dominant = counts.most_common(1)[0]
        boundary_body = (
            f"最终决策背景由后端逐点预测生成，共覆盖 {total} 个网格点。"
            f"面积占比最高的类别是 {dominant[0]}，约占 {dominant[1] / max(total, 1):.1%}。"
            f"{_boundary_comment(algorithm_id)}"
        )
    else:
        boundary_body = "当前最终帧没有背景网格，结果主要通过样本点和辅助几何对象解释。"

    x_name = data.feature_names[data.feature_indices[0]]
    y_name = data.feature_names[data.feature_indices[1]]
    return [
        {"title": "最终结果", "body": result_body},
        {"title": "边界形态", "body": boundary_body},
        {
            "title": "过程回顾",
            "body": (
                f"本次后端共生成 {len(steps)} 个状态帧，覆盖从初始状态、中间执行到最终决策的完整过程。"
                f"所有距离、投影、切线、权重、背景网格都在后端基于同一二维空间计算完成。"
            ),
        },
        {
            "title": "结果限定",
            "body": (
                f"当前分析只针对所选二维特征 {x_name} 与 {y_name}。"
                "如果切换特征或超参数，执行过程和最终边界都会重新计算，结果解释也会随之变化。"
            ),
        },
    ]


def _boundary_comment(algorithm_id: str) -> str:
    comments = {
        "knn": " KNN 的区域形态由局部邻居投票决定，K 越小越容易出现破碎边界。",
        "logistic-regression": " 逻辑回归边界保持线性，多分类时由多条两两等分线围成多边形区域。",
        "lda": " LDA 的最终背景来自一维投影轴上的阈值切分，因此边界仍然是线性的。",
        "decision-tree": " 决策树只做水平或垂直切分，因此最终区域呈正交矩形块状。",
        "adaboost": " AdaBoost 由多个树桩叠加投票，边界会呈现由横竖切线组合出的锯齿感。",
        "mlp": " MLP 通过隐藏层非线性变换形成平滑弯曲的决策区域。",
    }
    return comments.get(algorithm_id, "")


def _step(
    description: str,
    scatter: list[dict[str, Any]],
    background_grid: dict[str, Any] | None,
    helpers: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "index": 0,
        "description": description,
        "scatter": scatter,
        "backgroundGrid": background_grid,
        "helpers": helpers,
        "annotations": annotations or [],
    }


def _renumber_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, step in enumerate(steps):
        step["index"] = index
    return steps


def _bounds_for(points: np.ndarray) -> dict[str, float]:
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_pad = max((x_max - x_min) * 0.18, 0.8)
    y_pad = max((y_max - y_min) * 0.18, 0.8)
    return {
        "xMin": _num(x_min - x_pad),
        "xMax": _num(x_max + x_pad),
        "yMin": _num(y_min - y_pad),
        "yMax": _num(y_max + y_pad),
    }


def _scatter(
    data: PreparedRun,
    weights: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
    misclassified: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    return _scatter_from_arrays(data.x, data.y, data.labels, weights, predictions, misclassified)


def _scatter_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    weights: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
    misclassified: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if weights is None:
        weights = np.ones(len(x)) / max(len(x), 1)
    relative = weights / max(float(np.mean(weights)), 1e-12)
    points = []
    for index, (point, label) in enumerate(zip(x, y)):
        label_index = int(label)
        radius = 5.8 + min(13.0, max(0.0, math.sqrt(float(relative[index])) - 1.0) * 5.2)
        item = {
            "id": f"sample-{index}",
            "sourceIndex": index,
            "x": _num(point[0]),
            "y": _num(point[1]),
            "labelIndex": label_index,
            "labelName": labels[label_index],
            "color": PALETTE[label_index % len(PALETTE)],
            "weight": _num(float(weights[index])),
            "radius": _num(radius),
            "enlarged": bool(relative[index] > 1.35),
            "selected": False,
            "misclassified": bool(misclassified[index]) if misclassified is not None else False,
        }
        if predictions is not None:
            pred_index = int(predictions[index])
            item["predictedLabelIndex"] = pred_index
            item["predictedLabelName"] = labels[pred_index]
            item["predictedColor"] = PALETTE[pred_index % len(PALETTE)]
        points.append(item)
    return points


def _unknown_point(point: np.ndarray, item_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "sourceIndex": None,
        "x": _num(point[0]),
        "y": _num(point[1]),
        "labelIndex": -1,
        "labelName": "未知样本",
        "color": "#111827",
        "weight": 1.0,
        "radius": 11,
        "enlarged": True,
        "selected": True,
        "misclassified": False,
    }


def _grid(
    bounds: dict[str, float],
    predictor: Callable[[np.ndarray], np.ndarray],
    labels: list[str],
    resolution: int = 64,
) -> dict[str, Any]:
    x_values = np.linspace(bounds["xMin"], bounds["xMax"], resolution)
    y_values = np.linspace(bounds["yMin"], bounds["yMax"], resolution)
    xx, yy = np.meshgrid(x_values, y_values)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    predictions = np.asarray(predictor(flat), dtype=int)
    return {
        "columns": resolution,
        "rows": resolution,
        "xValues": [_num(value) for value in x_values],
        "yValues": [_num(value) for value in y_values],
        "points": [
            {
                "x": _num(point[0]),
                "y": _num(point[1]),
                "labelIndex": int(label),
                "labelName": labels[int(label)],
                "color": PALETTE[int(label) % len(PALETTE)],
            }
            for point, label in zip(flat, predictions)
        ],
    }


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _logistic_boundary_helpers(weights: np.ndarray, bounds: dict[str, float], labels: list[str]) -> list[dict[str, Any]]:
    helpers = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            diff = weights[:, left] - weights[:, right]
            line = _line_from_coefficients(diff[0], diff[1], diff[2], bounds)
            if line:
                line.update(
                    {
                        "color": PALETTE[left % len(PALETTE)],
                        "width": 2.0,
                        "label": f"{labels[left]} = {labels[right]}",
                    }
                )
                helpers.append(line)
    return helpers


def _logistic_error_segments(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    labels: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    scores = np.column_stack([x, np.ones(len(x))]) @ weights
    order = np.argsort(-np.max(scores, axis=1))[:limit]
    helpers = []
    for index in order:
        true_label = int(y[index])
        rival_scores = scores[index].copy()
        rival_scores[true_label] = -np.inf
        rival = int(np.argmax(rival_scores))
        diff = weights[:, true_label] - weights[:, rival]
        a, b, c = float(diff[0]), float(diff[1]), float(diff[2])
        denom = a * a + b * b
        if denom <= 1e-12:
            continue
        value = (a * x[index][0] + b * x[index][1] + c) / denom
        foot = np.array([x[index][0] - a * value, x[index][1] - b * value])
        helpers.append(
            {
                "type": "segment",
                "x1": _num(x[index][0]),
                "y1": _num(x[index][1]),
                "x2": _num(foot[0]),
                "y2": _num(foot[1]),
                "color": "rgba(15,23,42,0.26)",
                "width": 1,
                "dash": [4, 5],
                "label": f"{labels[true_label]} 到边界垂线",
            }
        )
    return helpers


def _lda_projection_helpers(
    x: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    y: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    helpers = []
    for index, point in enumerate(x):
        scalar = float((point - origin) @ axis)
        projection = origin + scalar * axis
        helpers.append(
            {
                "type": "segment",
                "x1": _num(point[0]),
                "y1": _num(point[1]),
                "x2": _num(projection[0]),
                "y2": _num(projection[1]),
                "color": "rgba(15,23,42,0.30)",
                "width": 1,
                "dash": [3, 5],
                "label": f"{labels[int(y[index])]} 正交投影",
            }
        )
        helpers.append(
            {
                "type": "point",
                "x": _num(projection[0]),
                "y": _num(projection[1]),
                "radius": 3.6,
                "color": PALETTE[int(y[index]) % len(PALETTE)],
                "label": "投影点",
            }
        )
    return helpers


def _lda_threshold_helpers(
    origin: np.ndarray,
    axis: np.ndarray,
    thresholds: list[float],
    bounds: dict[str, float],
) -> list[dict[str, Any]]:
    helpers = []
    normal = np.array([-axis[1], axis[0]])
    for index, threshold in enumerate(thresholds):
        point = origin + threshold * axis
        line = _line_from_point_direction(point, normal, bounds)
        line.update(
            {
                "color": "#0f172a",
                "width": 1.7,
                "dash": [8, 5],
                "label": f"LDA 阈值 {index + 1}",
            }
        )
        helpers.append(line)
        helpers.append(
            {
                "type": "point",
                "x": _num(point[0]),
                "y": _num(point[1]),
                "radius": 6,
                "color": "#0f172a",
                "label": f"阈值 {index + 1}",
            }
        )
    return helpers


def _collect_tree_splits(
    model: DecisionTreeClassifier,
    node_id: int,
    region: dict[str, float],
    splits: list[dict[str, Any]],
    labels: list[str],
) -> None:
    tree = model.tree_
    left = int(tree.children_left[node_id])
    right = int(tree.children_right[node_id])
    if left == right:
        return
    feature = int(tree.feature[node_id])
    threshold = float(tree.threshold[node_id])
    values = tree.value[node_id][0]
    predicted = int(np.argmax(values))
    split = {
        "nodeId": node_id,
        "feature": feature,
        "featureName": "X 特征" if feature == 0 else "Y 特征",
        "threshold": threshold,
        "region": region.copy(),
        "prediction": labels[predicted],
    }
    splits.append(split)
    left_region = region.copy()
    right_region = region.copy()
    if feature == 0:
        left_region["xMax"] = min(left_region["xMax"], threshold)
        right_region["xMin"] = max(right_region["xMin"], threshold)
    else:
        left_region["yMax"] = min(left_region["yMax"], threshold)
        right_region["yMin"] = max(right_region["yMin"], threshold)
    _collect_tree_splits(model, left, left_region, splits, labels)
    _collect_tree_splits(model, right, right_region, splits, labels)


def _tree_split_helper(split: dict[str, Any], bounds: dict[str, float]) -> dict[str, Any]:
    region = split["region"]
    if split["feature"] == 0:
        return {
            "type": "segment",
            "x1": _num(split["threshold"]),
            "y1": _num(region["yMin"]),
            "x2": _num(split["threshold"]),
            "y2": _num(region["yMax"]),
            "color": "#0f172a",
            "width": 2.4,
            "label": f"x <= {_num(split['threshold'])}",
        }
    return {
        "type": "segment",
        "x1": _num(region["xMin"]),
        "y1": _num(split["threshold"]),
        "x2": _num(region["xMax"]),
        "y2": _num(split["threshold"]),
        "color": "#0f172a",
        "width": 2.4,
        "label": f"y <= {_num(split['threshold'])}",
    }


def _stump_helper(
    stump: DecisionTreeClassifier,
    bounds: dict[str, float],
    color: str = "#dc2626",
    width: float = 2.5,
) -> dict[str, Any]:
    feature = int(stump.tree_.feature[0])
    threshold = float(stump.tree_.threshold[0])
    if feature < 0:
        return {
            "type": "segment",
            "x1": bounds["xMin"],
            "y1": bounds["yMin"],
            "x2": bounds["xMax"],
            "y2": bounds["yMax"],
            "color": color,
            "width": width,
            "dash": [6, 6],
            "label": "弱分类器未产生有效切分",
        }
    if feature == 0:
        return {
            "type": "segment",
            "x1": _num(threshold),
            "y1": bounds["yMin"],
            "x2": _num(threshold),
            "y2": bounds["yMax"],
            "color": color,
            "width": width,
            "label": f"弱分类器：x <= {_num(threshold)}",
        }
    return {
        "type": "segment",
        "x1": bounds["xMin"],
        "y1": _num(threshold),
        "x2": bounds["xMax"],
        "y2": _num(threshold),
        "color": color,
        "width": width,
        "label": f"弱分类器：y <= {_num(threshold)}",
    }


def _line_from_point_direction(point: np.ndarray, direction: np.ndarray, bounds: dict[str, float]) -> dict[str, Any]:
    normal = np.array([-direction[1], direction[0]])
    c = -float(normal @ point)
    line = _line_from_coefficients(float(normal[0]), float(normal[1]), c, bounds)
    if line is None:
        return {
            "type": "segment",
            "x1": _num(point[0] - direction[0]),
            "y1": _num(point[1] - direction[1]),
            "x2": _num(point[0] + direction[0]),
            "y2": _num(point[1] + direction[1]),
            "color": "#111827",
            "width": 2,
        }
    line.update({"color": "#111827", "width": 2.4, "label": "判别轴"})
    return line


def _line_from_coefficients(a: float, b: float, c: float, bounds: dict[str, float]) -> dict[str, Any] | None:
    points = []
    for x_value in (bounds["xMin"], bounds["xMax"]):
        if abs(b) > 1e-12:
            y_value = (-a * x_value - c) / b
            if bounds["yMin"] - 1e-9 <= y_value <= bounds["yMax"] + 1e-9:
                points.append((x_value, y_value))
    for y_value in (bounds["yMin"], bounds["yMax"]):
        if abs(a) > 1e-12:
            x_value = (-b * y_value - c) / a
            if bounds["xMin"] - 1e-9 <= x_value <= bounds["xMax"] + 1e-9:
                points.append((x_value, y_value))
    unique = []
    for point in points:
        rounded = (_num(point[0]), _num(point[1]))
        if rounded not in unique:
            unique.append(rounded)
    if len(unique) < 2:
        return None
    return {
        "type": "segment",
        "x1": unique[0][0],
        "y1": unique[0][1],
        "x2": unique[1][0],
        "y2": unique[1][1],
    }


def _hidden_layers(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parsed = [int(item) for item in value]
    else:
        parsed = []
        for item in str(value).replace("，", ",").split(","):
            item = item.strip()
            if item:
                parsed.append(int(float(item)))
    parsed = [min(max(item, 2), 128) for item in parsed[:3]]
    return tuple(parsed or [32, 16])


def _int_param(params: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(params.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def _float_param(params: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def _num(value: float | int) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return round(value, 5)
