from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import OptimizeWarning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core.datasets import get_dataset_summary, load_dataset
from core.frame_algorithms import ALGORITHMS

warnings.filterwarnings("ignore", category=OptimizeWarning)


def evaluate_algorithm(
    algorithm_id: str,
    dataset_id: str,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = hyperparameters or {}
    loaded = load_dataset(dataset_id)
    labels = loaded["targetNames"]
    x = StandardScaler().fit_transform(loaded["data"])
    y = np.asarray(loaded["target"], dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=_float_param(params, "testSize", 0.28, 0.15, 0.45),
        random_state=_int_param(params, "randomSeed", 42, 0, 100000),
        stratify=y,
    )
    model = _build_model(algorithm_id, params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    train_predictions = model.predict(x_train)
    probabilities = _predict_probabilities(model, x_test)

    algorithm = ALGORITHMS[algorithm_id]
    matrix = confusion_matrix(y_test, predictions, labels=list(range(len(labels))))
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, predictions)
    train_accuracy = accuracy_score(y_train, train_predictions)
    balanced = balanced_accuracy_score(y_test, predictions)

    return {
        "algorithmId": algorithm_id,
        "algorithmName": algorithm["name"],
        "dataset": get_dataset_summary(dataset_id, include_preview=True),
        "parameters": _public_parameters(algorithm_id, params),
        "metrics": [
            {"label": "Accuracy", "value": _num(accuracy), "format": "percent"},
            {"label": "Train Accuracy", "value": _num(train_accuracy), "format": "percent"},
            {"label": "Generalization Gap", "value": _num(train_accuracy - accuracy), "format": "percent"},
            {"label": "Balanced Accuracy", "value": _num(balanced), "format": "percent"},
            {"label": "Macro Precision", "value": _num(macro_precision), "format": "percent"},
            {"label": "Macro Recall", "value": _num(macro_recall), "format": "percent"},
            {"label": "Macro F1", "value": _num(macro_f1), "format": "percent"},
            {"label": "Weighted Precision", "value": _num(weighted_precision), "format": "percent"},
            {"label": "Weighted Recall", "value": _num(weighted_recall), "format": "percent"},
            {"label": "Weighted F1", "value": _num(weighted_f1), "format": "percent"},
            {"label": "训练样本", "value": int(len(y_train)), "format": "integer"},
            {"label": "测试样本", "value": int(len(y_test)), "format": "integer"},
        ],
        "charts": {
            "confusionMatrix": {
                "labels": labels,
                "values": matrix.astype(int).tolist(),
            },
            "classMetrics": [
                {
                    "label": label,
                    "precision": _num(class_precision[index]),
                    "recall": _num(class_recall[index]),
                    "f1": _num(class_f1[index]),
                    "support": int(class_support[index]),
                }
                for index, label in enumerate(labels)
            ],
            "predictions": _prediction_rows(labels, y_test, predictions, probabilities),
        },
        "analysis": _analysis(
            algorithm_id,
            algorithm["name"],
            loaded["config"]["name"],
            labels,
            matrix,
            accuracy,
            train_accuracy,
            balanced,
        ),
    }


def _build_model(algorithm_id: str, params: dict[str, Any]):
    if algorithm_id == "knn":
        return KNeighborsClassifier(
            n_neighbors=_int_param(params, "k", 5, 1, 35),
            metric=str(params.get("distanceMetric", "euclidean")),
        )
    if algorithm_id == "logistic-regression":
        l2 = _float_param(params, "l2", 0.001, 0.0, 2.0)
        return LogisticRegression(
            C=1.0 / max(l2, 1e-6),
            max_iter=_int_param(params, "iterations", 260, 30, 1000),
            random_state=_int_param(params, "randomSeed", 42, 0, 100000),
        )
    if algorithm_id == "lda":
        return LinearDiscriminantAnalysis()
    if algorithm_id == "decision-tree":
        criterion = str(params.get("criterion", "gini"))
        if criterion not in {"gini", "entropy", "log_loss"}:
            criterion = "gini"
        return DecisionTreeClassifier(
            criterion=criterion,
            max_depth=_int_param(params, "maxDepth", 4, 1, 16),
            min_samples_leaf=_int_param(params, "minSamplesLeaf", 3, 1, 50),
            random_state=_int_param(params, "randomSeed", 42, 0, 100000),
        )
    if algorithm_id == "adaboost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=_int_param(params, "nEstimators", 40, 1, 200),
            learning_rate=_float_param(params, "learningRate", 0.8, 0.01, 3.0),
            random_state=_int_param(params, "randomSeed", 42, 0, 100000),
        )
    if algorithm_id == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=_hidden_layers(params.get("hiddenLayers", "32,16")),
            activation=str(params.get("activation", "tanh")),
            solver="adam",
            alpha=_float_param(params, "alpha", 0.0003, 0.0, 1.0),
            learning_rate_init=_float_param(params, "learningRate", 0.025, 0.0001, 1.0),
            max_iter=_int_param(params, "epochs", 140, 10, 800),
            random_state=_int_param(params, "randomSeed", 42, 0, 100000),
        )
    raise ValueError("Unsupported algorithm")


def _predict_probabilities(model: Any, x_test: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)
    return None


def _prediction_rows(
    labels: list[str],
    y_test: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows = []
    for index, (actual, predicted) in enumerate(zip(y_test[:24], predictions[:24])):
        confidence = None
        if probabilities is not None:
            confidence = _num(float(np.max(probabilities[index])))
        rows.append(
            {
                "sample": index + 1,
                "actual": labels[int(actual)],
                "predicted": labels[int(predicted)],
                "correct": bool(int(actual) == int(predicted)),
                "confidence": confidence,
            }
        )
    return rows


def _analysis(
    algorithm_id: str,
    algorithm_name: str,
    dataset_name: str,
    labels: list[str],
    matrix: np.ndarray,
    accuracy: float,
    train_accuracy: float,
    balanced: float,
) -> list[dict[str, str]]:
    recalls = []
    for index, label in enumerate(labels):
        total = max(int(np.sum(matrix[index])), 1)
        recalls.append((label, matrix[index][index] / total))
    weakest = min(recalls, key=lambda item: item[1])
    gap = train_accuracy - accuracy
    confused_pairs = _confused_pairs(labels, matrix)
    behavior = _algorithm_behavior(algorithm_id)
    gap_message = _gap_message(gap)
    confusion_message = (
        f"最明显的混淆方向是 {confused_pairs[0]}。"
        if confused_pairs
        else "混淆矩阵中没有明显的非对角线错误，说明当前划分下各类别区分较稳定。"
    )
    return [
        {
            "title": "总体表现",
            "body": f"{algorithm_name} 在 {dataset_name} 的测试集准确率为 {accuracy:.2%}，平衡准确率为 {balanced:.2%}。Accuracy 看总体答对比例，Balanced Accuracy 会平均看每个类别的召回率，类别样本不均衡时后者更值得参考。",
        },
        {
            "title": "类别表现",
            "body": f"当前召回率最低的类别是 {weakest[0]}，召回率约为 {weakest[1]:.2%}。召回率低表示该类别中有较多样本被分到别的类别，需要结合混淆矩阵判断它主要被谁“吸走”。",
        },
        {
            "title": "混淆矩阵解读",
            "body": f"{confusion_message} 混淆通常发生在特征分布接近、类别边界重叠或测试样本位于边缘区域时。",
        },
        {
            "title": "泛化判断",
            "body": f"训练准确率与测试准确率差距为 {gap:.2%}。{gap_message}",
        },
        {
            "title": "算法特性解释",
            "body": behavior["reading"],
        },
        {
            "title": "调参方向",
            "body": behavior["tuning"],
        },
        {
            "title": "与可视化模块的关系",
            "body": "这里使用真实数据集和全部特征评估预测表现；独立可视化模块使用二维合成数据解释算法执行过程。前者回答“预测得怎么样”，后者回答“算法是怎么一步步做决定的”。",
        },
    ]


def _confused_pairs(labels: list[str], matrix: np.ndarray) -> list[str]:
    pairs = []
    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if actual_index == predicted_index:
                continue
            count = int(matrix[actual_index][predicted_index])
            if count > 0:
                pairs.append((count, f"{actual_label} 被预测为 {predicted_label}：{count} 个"))
    return [text for _, text in sorted(pairs, reverse=True)[:3]]


def _gap_message(gap: float) -> str:
    if gap > 0.12:
        return "这个差距偏大，说明模型明显更适应训练集，后续应优先降低模型复杂度、增强正则或重新检查数据划分。"
    if gap > 0.05:
        return "这个差距处于需要关注的范围，模型可能有轻微过拟合，建议结合混淆矩阵和类别指标再判断。"
    if gap < -0.03:
        return "测试准确率高于训练准确率，通常来自数据划分波动或测试集刚好更容易，不应简单理解为模型一定更强。"
    return "这个差距较小，说明当前划分下训练表现和测试表现比较一致，泛化状态相对稳定。"


def _algorithm_behavior(algorithm_id: str) -> dict[str, str]:
    content = {
        "knn": {
            "reading": "KNN 的表现主要由局部邻域决定。如果错误集中在类别交界处，通常说明附近邻居类别混杂；如果某一类召回率低，可能是该类样本密度不足或被多数类邻居包围。",
            "tuning": "可以尝试调节 K 值和距离度量。K 小会更敏感、更局部，K 大会更平滑、更稳健；若特征尺度不同，必须保持标准化，否则距离排序会失真。",
        },
        "logistic-regression": {
            "reading": "逻辑回归学习的是线性概率边界。若总体准确率不错但个别类别混淆明显，通常说明这些类别在当前特征空间中只能被近似线性分开。",
            "tuning": "可以增加迭代次数确保收敛，或调整 L2 正则强度。若错误来自明显非线性结构，单纯调参收益有限，需要引入特征变换或换用非线性模型。",
        },
        "lda": {
            "reading": "LDA 依赖类均值和共享协方差假设。它表现好时，说明类别中心和类内散布已经提供了足够判别信息；表现差时，多半是类别协方差差异大或边界并非线性。",
            "tuning": "LDA 可调空间较小，重点应放在特征质量、异常值处理和类别分布检查上。如果混淆集中在形态相近的类别，可与逻辑回归、树模型对比。",
        },
        "decision-tree": {
            "reading": "决策树用一组正交规则解释预测。训练准确率高但测试准确率低时，常见原因是树把训练集切得太细，学到了偶然噪声。",
            "tuning": "优先调整最大深度、叶子最小样本数和切分准则。想提升稳定性时，通常会从单棵树走向随机森林或提升树等集成方法。",
        },
        "adaboost": {
            "reading": "AdaBoost 会持续关注错分样本。若错误样本是真正困难边界点，它能逐步修补；若错误来自噪声或异常值，模型可能不断放大这些点的影响。",
            "tuning": "可以调节弱分类器数量和学习率。弱分类器太少会欠拟合，太多或学习率太高可能追逐噪声；当指标进入平台期后继续增加轮数收益通常有限。",
        },
        "mlp": {
            "reading": "MLP 具备较强非线性表达能力。若训练和测试都好，说明网络捕捉到了有效结构；若训练好测试差，往往是隐藏层规模、训练轮数或学习率让模型过度贴合训练集。",
            "tuning": "可以调整隐藏层规模、激活函数、学习率、Alpha 正则和 Epoch。MLP 对标准化很敏感，也更需要关注泛化差距而不只是最终准确率。",
        },
    }
    return content[algorithm_id]


def _public_parameters(algorithm_id: str, params: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "knn": {"k": 5, "distanceMetric": "euclidean", "testSize": 0.28, "randomSeed": 42},
        "logistic-regression": {"iterations": 260, "l2": 0.001, "testSize": 0.28, "randomSeed": 42},
        "lda": {"regularization": 0.001, "testSize": 0.28, "randomSeed": 42},
        "decision-tree": {"criterion": "gini", "maxDepth": 4, "minSamplesLeaf": 3, "testSize": 0.28, "randomSeed": 42},
        "adaboost": {"nEstimators": 40, "learningRate": 0.8, "testSize": 0.28, "randomSeed": 42},
        "mlp": {"hiddenLayers": "32,16", "activation": "tanh", "learningRate": 0.025, "alpha": 0.0003, "epochs": 140, "testSize": 0.28, "randomSeed": 42},
    }
    merged = defaults[algorithm_id].copy()
    merged.update(params)
    return merged


def _hidden_layers(value: Any) -> tuple[int, ...]:
    parsed = []
    for item in str(value).replace("，", ",").split(","):
        item = item.strip()
        if item:
            parsed.append(min(max(int(float(item)), 2), 128))
    return tuple(parsed[:3] or [32, 16])


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
    return round(float(value), 5)
