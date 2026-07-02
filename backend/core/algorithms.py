from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import OptimizeWarning
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from core.datasets import get_dataset_summary, load_dataset


PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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
                "body": "KNN 是一种基于实例的分类方法。训练阶段几乎不做参数学习，而是保留训练样本；预测阶段计算待分类样本与所有训练样本之间的距离，再从最近的 K 个邻居中进行多数投票。",
            },
            {
                "title": "运行流程",
                "body": "系统会先把真实数据集划分为训练集和测试集，再将标准化后的高维特征投影到二维画布中。训练集作为已知样本，测试集作为待预测样本，KNN 会计算测试样本到训练样本的距离。",
            },
            {
                "title": "关键参数",
                "body": "K 值控制局部邻域大小。K 太小会让边界过度贴合局部噪声；K 太大则会弱化局部结构，使少数类样本更容易被多数类淹没。距离度量和特征缩放也会直接影响邻居排序。",
            },
            {
                "title": "观察重点",
                "body": "重点观察未知点附近是否存在类别混杂、最近邻距离是否接近、K 值变化是否导致分类翻转。如果分类结果对 K 值非常敏感，通常说明该区域处于类别边界或样本密度不足。",
            },
            {
                "title": "局限性",
                "body": "KNN 在样本量大或维度高时预测成本较高，并且容易受到无关特征和尺度差异影响。因此教学演示中必须先做标准化，并结合准确率曲线理解 K 值选择。",
            },
        ],
    },
    "logistic-regression": {
        "id": "logistic-regression",
        "name": "逻辑回归",
        "short_name": "Logistic Regression",
        "category": "传统统计与距离模型",
        "summary": "用线性函数生成分类得分，再通过 Sigmoid 或 Softmax 转换为概率。",
        "principle": [
            {
                "title": "核心思想",
                "body": "逻辑回归用线性函数计算分类得分，再通过 Sigmoid 或 Softmax 将得分转换为概率。它虽然名字中包含回归，但在分类任务中学习的是类别边界。",
            },
            {
                "title": "训练目标",
                "body": "模型通过最小化对数损失来调整特征权重。预测正确且置信度高时损失较低；预测错误或置信度不足时损失较高，因此 loss 曲线可以反映概率模型是否收敛。",
            },
            {
                "title": "边界含义",
                "body": "在线性可分或近似线性可分的数据上，逻辑回归通常表现稳定。二维画布中的背景色代表模型更倾向的类别区域，颜色交界处就是分类边界。",
            },
            {
                "title": "观察重点",
                "body": "重点观察样本点是否主要分布在对应类别区域内、边界附近是否存在大量混杂点、概率置信度是否过低。如果边界无法贴合数据形状，说明线性假设不足。",
            },
            {
                "title": "局限性",
                "body": "逻辑回归的基础形式表达能力有限，难以直接拟合复杂非线性边界。它的优势在于速度快、概率输出清晰、权重具有一定解释性。",
            },
        ],
    },
    "lda": {
        "id": "lda",
        "name": "线性判别分析",
        "short_name": "LDA",
        "category": "传统统计与距离模型",
        "summary": "寻找能最大化类间距离并最小化类内方差的判别投影方向。",
        "principle": [
            {
                "title": "核心思想",
                "body": "LDA 寻找一组投影方向，使同一类别样本尽量聚集，不同类别样本尽量分开。它关注的是判别能力，而不是单纯保留最大方差。",
            },
            {
                "title": "统计假设",
                "body": "LDA 会估计每个类别的均值和整体协方差，并假设不同类别共享相近的协方差结构。这个假设成立时，LDA 往往能得到非常清晰的线性判别空间。",
            },
            {
                "title": "投影过程",
                "body": "画布可以观察原始二维投影与 LDA 判别投影之间的变化。若投影后类别中心距离变大、同类点云更紧凑，说明判别方向有效。",
            },
            {
                "title": "观察重点",
                "body": "重点观察类间距离、类内散布和混淆矩阵。LDA 不一定追求视觉上最大展开，而是追求对分类最有用的方向。",
            },
            {
                "title": "局限性",
                "body": "当类别边界明显非线性、协方差差异很大或异常值较多时，LDA 的线性判别假设可能不足，需要与非线性模型对比。",
            },
        ],
    },
    "decision-tree": {
        "id": "decision-tree",
        "name": "决策树",
        "short_name": "Decision Tree",
        "category": "树模型、集成学习与深度学习",
        "summary": "基于信息增益或基尼系数递归划分特征空间，形成可解释的树状规则。",
        "principle": [
            {
                "title": "核心思想",
                "body": "决策树通过一系列 if-then 规则递归划分特征空间。每次分裂都会选择能让子节点类别更纯的特征和阈值，常用标准包括基尼系数和信息增益。",
            },
            {
                "title": "层级结构",
                "body": "根节点代表最先使用的规则，越靠近根节点的特征通常越重要。叶节点给出最终类别预测，路径上的每个判断共同构成一条可解释规则。",
            },
            {
                "title": "复杂度控制",
                "body": "树越深越能贴合训练集，但也更容易学习噪声。限制最大深度、最小叶子样本数或进行剪枝，可以降低过拟合风险。",
            },
            {
                "title": "观察重点",
                "body": "画布中的轴向切分展示了树模型如何把空间切成矩形区域。重点观察切分是否过细、叶子数量是否过多，以及测试集准确率是否低于训练集。",
            },
            {
                "title": "局限性",
                "body": "单棵树对数据扰动敏感，边界呈阶梯状，不擅长表达平滑边界。它的强项是解释性，弱项是稳定性和泛化能力。",
            },
        ],
    },
    "adaboost": {
        "id": "adaboost",
        "name": "AdaBoost 集成学习",
        "short_name": "AdaBoost",
        "category": "树模型、集成学习与深度学习",
        "summary": "串行训练多个弱分类器，提高前一轮错分样本权重并加权投票。",
        "principle": [
            {
                "title": "核心思想",
                "body": "AdaBoost 串行训练多个弱分类器。每一轮都会提高上一轮错分样本的权重，使后续分类器更关注难分样本，最终通过加权投票形成强分类器。",
            },
            {
                "title": "样本权重",
                "body": "画布中点的大小可以表示当前迭代轮次下的关注程度。被错分或靠近边界的样本会变得更醒目，体现模型正在把注意力转向困难区域。",
            },
            {
                "title": "集成过程",
                "body": "早期弱分类器通常快速提升准确率，后续分类器负责修补局部错误。阶段准确率曲线可以帮助判断新增弱分类器是否仍然有效。",
            },
            {
                "title": "观察重点",
                "body": "重点观察权重是否集中在少数异常点、阶段准确率是否进入平台期、弱分类器权重是否出现明显下降。这些现象会影响继续迭代的收益。",
            },
            {
                "title": "局限性",
                "body": "AdaBoost 对异常值较敏感，因为异常样本会持续获得较高权重。如果数据噪声较多，过多迭代可能让模型把精力消耗在不可解释的点上。",
            },
        ],
    },
    "mlp": {
        "id": "mlp",
        "name": "多层感知机",
        "short_name": "MLP",
        "category": "树模型、集成学习与深度学习",
        "summary": "通过多层神经元、非线性激活和反向传播学习复杂分类边界。",
        "principle": [
            {
                "title": "核心思想",
                "body": "MLP 由输入层、隐藏层和输出层组成。每一层都对上一层输出做加权组合，再经过非线性激活函数，从而逐步构造更复杂的特征表达。",
            },
            {
                "title": "训练过程",
                "body": "模型通过前向传播得到预测结果，再通过反向传播计算误差对权重的影响，最后使用优化器更新参数。训练 loss 曲线反映了这个优化过程是否稳定。",
            },
            {
                "title": "非线性边界",
                "body": "隐藏层让 MLP 能学习弯曲、组合和局部变化的分类边界。画布中的背景区域可以帮助观察模型是否捕捉到类别形状，而不只是画直线。",
            },
            {
                "title": "观察重点",
                "body": "重点观察 loss 是否持续下降、边界是否过度扭曲、训练集和测试集指标是否出现明显差距。指标差距过大时，通常意味着模型开始过拟合。",
            },
            {
                "title": "局限性",
                "body": "MLP 的可解释性弱于线性模型和决策树，对超参数、特征缩放和训练轮数更敏感。教学中应把它作为非线性能力的代表，而不是默认最优解。",
            },
        ],
    },
}


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_train_2d: np.ndarray
    x_test_2d: np.ndarray
    x_all_2d: np.ndarray
    y_all: np.ndarray
    x_all_scaled: np.ndarray
    labels: list[str]
    feature_names: list[str]
    scaler: StandardScaler


def run_algorithm(algorithm_id: str, dataset_id: str) -> dict[str, Any]:
    prepared = _prepare_data(dataset_id)
    runners = {
        "knn": _run_knn,
        "logistic-regression": _run_logistic_regression,
        "lda": _run_lda,
        "decision-tree": _run_decision_tree,
        "adaboost": _run_adaboost,
        "mlp": _run_mlp,
    }
    result = runners[algorithm_id](prepared)
    algorithm = ALGORITHMS[algorithm_id]
    return {
        "algorithm": algorithm,
        "dataset": get_dataset_summary(dataset_id, include_preview=True),
        "charts": result["charts"],
        "metrics": result["metrics"],
        "analysis": result["analysis"],
    }


def _prepare_data(dataset_id: str) -> PreparedData:
    loaded = load_dataset(dataset_id)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(loaded["target"])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(loaded["data"])
    pca = PCA(n_components=2, random_state=42)
    x_2d = pca.fit_transform(x_scaled)
    x_train, x_test, y_train, y_test, x_train_2d, x_test_2d = train_test_split(
        x_scaled,
        y,
        x_2d,
        test_size=0.28,
        random_state=42,
        stratify=y,
    )
    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_train_2d=x_train_2d,
        x_test_2d=x_test_2d,
        x_all_2d=x_2d,
        y_all=y,
        x_all_scaled=x_scaled,
        labels=loaded["targetNames"],
        feature_names=loaded["featureNames"],
        scaler=scaler,
    )


def _run_knn(data: PreparedData) -> dict[str, Any]:
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    accuracy = accuracy_score(data.y_test, predictions)
    unknown = data.x_test[0:1]
    distances, indices = model.kneighbors(unknown, n_neighbors=5)
    neighbor_points = [
        {
            "point": _point(data.x_train_2d[index], int(data.y_train[index]), data.labels),
            "distance": round(float(distances[0][rank]), 3),
            "rank": rank + 1,
        }
        for rank, index in enumerate(indices[0])
    ]
    k_curve = []
    for k in range(1, 16):
        candidate = KNeighborsClassifier(n_neighbors=k)
        candidate.fit(data.x_train, data.y_train)
        k_curve.append(
            {
                "k": k,
                "accuracy": round(float(candidate.score(data.x_test, data.y_test)), 4),
            }
        )
    return _standard_payload(
        data,
        model,
        predictions,
        accuracy,
        charts={
            "type": "knn",
            "scatter": _scatter(data.x_all_2d, data.y_all, data.labels),
            "unknown": _point(data.x_test_2d[0], int(data.y_test[0]), data.labels),
            "neighbors": neighbor_points,
            "kCurve": k_curve,
            "canvas": {
                "mode": "knn",
                **_knn_canvas(data, model),
                "steps": [
                    {"title": "展示训练集", "detail": "训练集样本是已知类别的历史数据，KNN 会保存这些样本。"},
                    {"title": "选择测试样本", "detail": "测试集样本是真实数据集中留出的样本，暂时视为未知类别。"},
                    {"title": "计算距离", "detail": "计算测试样本到所有训练样本的距离，并按距离排序。"},
                    {"title": "选择最近邻", "detail": "按当前 K 值选择最近的训练样本作为投票邻居。"},
                    {"title": "多数投票", "detail": "统计最近邻类别票数，得到测试样本的预测类别。"},
                ],
            },
        },
        analysis=_analysis(
            "KNN 在该数据集上的判断主要依赖局部样本密度。当前默认 K=5 时，模型会综合最近五个样本的标签，避免单个噪声点直接决定预测结果。",
            "从画布上看，测试样本如果落在某个类别点云内部，连线通常会集中指向同一颜色；如果落在两个类别交界处，连线会跨越多个颜色区域，此时分类结果更容易随 K 值变化。",
            "评测时不要只看 Accuracy。Macro Recall 能反映少数类别是否被忽略，F1 能综合考虑误报和漏报。如果训练准确率显著高于测试准确率，说明 K 值可能过小或数据局部噪声较多。",
            "教学演示时建议切换不同测试样本，观察真实测试样本的最近邻集合如何变化。再结合 K 值曲线解释参数选择，而不是把某一个 K 值当成固定答案。",
        ),
    )


def _run_logistic_regression(data: PreparedData) -> dict[str, Any]:
    model = LogisticRegression(max_iter=500)
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    probabilities = model.predict_proba(data.x_test)
    accuracy = accuracy_score(data.y_test, predictions)
    loss_curve = []
    for iteration in range(20, 301, 20):
        candidate = LogisticRegression(max_iter=iteration)
        candidate.fit(data.x_train, data.y_train)
        candidate_probs = candidate.predict_proba(data.x_test)
        loss_curve.append(
            {
                "iteration": iteration,
                "loss": round(float(log_loss(data.y_test, candidate_probs)), 4),
            }
        )
    return _standard_payload(
        data,
        model,
        predictions,
        accuracy,
        charts={
            "type": "logistic-regression",
            "scatter": _scatter(data.x_all_2d, data.y_all, data.labels),
            "lossCurve": loss_curve,
            "canvas": {
                "mode": "boundary",
                **_split_canvas(data, model),
                "steps": [
                    {"title": "划分训练集与测试集", "detail": "训练集用于拟合权重，测试集暂不显示类别，用于检验预测效果。"},
                    {"title": "训练线性概率模型", "detail": "逻辑回归在训练集上学习特征权重和概率映射。"},
                    {"title": "预测测试集", "detail": "模型对每个测试样本输出预测类别和置信度。"},
                    {"title": "对照真实标签", "detail": "测试集按预测类别着色，预测错误的样本用橙色描边标出。"},
                ],
            },
            "probabilities": [
                {
                    "sample": index + 1,
                    "actual": data.labels[int(data.y_test[index])],
                    "predicted": data.labels[int(predictions[index])],
                    "confidence": round(float(np.max(probabilities[index])), 4),
                }
                for index in range(min(12, len(predictions)))
            ],
        },
        analysis=_analysis(
            "逻辑回归在该数据集上学习的是线性或近似线性的类别分割方式。它不会刻意弯曲边界去追逐局部噪声，因此在结构清晰的数据上通常表现稳定。",
            "画布背景区域展示了模型对二维投影空间的类别偏好。若样本点大多落在对应颜色区域，说明线性假设基本可用；若大量样本穿越颜色边界，说明特征空间中存在更复杂的非线性结构。",
            "Loss 曲线用于判断概率模型是否收敛。Accuracy 只能说明预测是否正确，而 Log Loss 会惩罚错误且过度自信的预测，因此更适合解释概率输出质量。",
            "结果解读时需要关注 Precision、Recall 和 F1 的平衡。如果某一类 Recall 偏低，说明模型经常漏掉该类；如果 Precision 偏低，则说明模型经常把其他类误判成该类。",
        ),
    )


def _run_lda(data: PreparedData) -> dict[str, Any]:
    model = LinearDiscriminantAnalysis()
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    accuracy = accuracy_score(data.y_test, predictions)
    projected = model.transform(np.vstack([data.x_train, data.x_test]))
    if projected.shape[1] == 1:
        projected_2d = np.column_stack([projected[:, 0], np.zeros(projected.shape[0])])
    else:
        projected_2d = projected[:, :2]
    combined_y = np.concatenate([data.y_train, data.y_test])
    return _standard_payload(
        data,
        model,
        predictions,
        accuracy,
        charts={
            "type": "lda",
            "before": _scatter(data.x_all_2d, data.y_all, data.labels),
            "projection": _scatter(projected_2d, combined_y, data.labels),
            "canvas": {
                "mode": "projection",
                **_lda_canvas(data, model),
                "steps": [
                    {"title": "展示原始二维投影", "detail": "先用 PCA 二维投影展示真实训练集和测试集位置。"},
                    {"title": "切换到 LDA 判别投影", "detail": "LDA 将样本投影到判别坐标，不绘制虚构连线或放射状结构。"},
                    {"title": "预测测试集", "detail": "测试集样本在 LDA 判别空间中按预测类别着色。"},
                    {"title": "对照真实标签", "detail": "预测错误的测试样本用橙色描边标出。"},
                ],
            },
        },
        analysis=_analysis(
            "LDA 的核心价值不只是分类，而是把数据投影到更有判别力的空间。与 PCA 关注总体方差不同，LDA 更关注类别标签带来的分离信息。",
            "如果投影后同类样本聚集、不同类别中心明显分开，说明数据满足 LDA 的线性判别假设。若投影后仍然有大量重叠，说明类别之间的边界可能不是简单线性的。",
            "混淆矩阵可以指出具体哪些类别互相混淆。对于三分类数据，某两个类别之间的错误集中出现，往往比整体 Accuracy 下降更值得分析。",
            "教学中可以对比 PCA 投影和 LDA 投影：PCA 可能让整体散布更大，但未必更利于分类；LDA 的目标是分类可分性，这一点需要通过画布动态投影过程体现。",
        ),
    )


def _run_decision_tree(data: PreparedData) -> dict[str, Any]:
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    accuracy = accuracy_score(data.y_test, predictions)
    unpruned = DecisionTreeClassifier(random_state=42)
    pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
    unpruned.fit(data.x_train, data.y_train)
    pruned.fit(data.x_train, data.y_train)
    comparison = [
        {
            "model": "未限制深度",
            "depth": int(unpruned.get_depth()),
            "leaves": int(unpruned.get_n_leaves()),
            "accuracy": round(float(unpruned.score(data.x_test, data.y_test)), 4),
        },
        {
            "model": "限制深度=3",
            "depth": int(pruned.get_depth()),
            "leaves": int(pruned.get_n_leaves()),
            "accuracy": round(float(pruned.score(data.x_test, data.y_test)), 4),
        },
    ]
    rules = export_text(model, feature_names=data.feature_names, max_depth=3).splitlines()
    return _standard_payload(
        data,
        model,
        predictions,
        accuracy,
        charts={
            "type": "decision-tree",
            "rules": rules[:24],
            "complexity": comparison,
            "featureImportance": _feature_importance(model, data.feature_names),
            "canvas": {
                "mode": "tree",
                **_split_canvas(data, model),
                "tree": _tree_nodes(model, data.feature_names, data.labels),
                "steps": [
                    {"title": "展示训练集与测试集", "detail": "训练集带真实标签，测试集暂时隐藏标签。"},
                    {"title": "训练决策树规则", "detail": "决策树在完整特征空间中学习分裂规则，而不是在画布上手工画线。"},
                    {"title": "预测测试集", "detail": "测试集通过树路径到达叶节点，并得到预测类别。"},
                    {"title": "对照真实标签", "detail": "测试集按预测类别着色，错误预测用橙色描边标出。"},
                ],
            },
        },
        analysis=_analysis(
            "决策树给出的不是平滑边界，而是一系列轴向切分规则。它的优势是每条预测路径都能追溯，适合教学中解释模型如何一步步做决策。",
            "复杂度对比是理解决策树的关键。未限制深度时，训练集通常表现更好，但叶节点数量增加会让模型记住局部噪声；限制深度后，边界更粗糙但泛化可能更稳定。",
            "特征重要性可以解释哪些字段最常用于有效分裂。它不能等同于因果关系，但能帮助观察模型主要依赖哪些测量维度。",
            "结果分析时要同时看训练准确率、测试准确率和泛化差距。如果测试集没有提升但树深度和叶子数量明显增加，说明模型复杂度已经超过当前数据需要。",
        ),
    )


def _run_adaboost(data: PreparedData) -> dict[str, Any]:
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=40,
        learning_rate=0.8,
        random_state=42,
    )
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    accuracy = accuracy_score(data.y_test, predictions)
    staged = [
        {"round": index + 1, "accuracy": round(float(score), 4)}
        for index, score in enumerate(model.staged_score(data.x_test, data.y_test))
    ]
    estimator_weights = [
        {
            "round": index + 1,
            "weight": round(float(weight), 4),
        }
        for index, weight in enumerate(model.estimator_weights_[:20])
    ]
    return _standard_payload(
        data,
        model,
        predictions,
        accuracy,
        charts={
            "type": "adaboost",
            "scatter": _scatter(data.x_all_2d, data.y_all, data.labels),
            "stagedAccuracy": staged,
            "estimatorWeights": estimator_weights,
            "canvas": {
                "mode": "boosting",
                **_split_canvas(data, model),
                "steps": [
                    {"title": "展示训练集与测试集", "detail": "训练集用于迭代训练弱分类器，测试集用于最终验证。"},
                    {"title": "训练弱分类器序列", "detail": "AdaBoost 在训练集上逐轮组合弱分类器。"},
                    {"title": "集成模型预测测试集", "detail": "测试集由最终加权集成模型给出预测类别。"},
                    {"title": "对照真实标签", "detail": "测试集按预测类别着色，错误预测用橙色描边标出。"},
                ],
            },
        },
        analysis=_analysis(
            "AdaBoost 的运行过程可以理解为不断重新分配注意力。早期模型先处理容易区分的样本，后续模型逐渐把重点放到错分样本和边界样本上。",
            "阶段准确率曲线反映了集成规模的收益。如果前几轮快速上升，说明弱分类器能够捕捉主要结构；如果后续进入平台期，继续增加轮数带来的收益会变小。",
            "弱分类器权重可以帮助判断每一轮模型是否仍有贡献。权重越高，说明该弱分类器在当前加权数据上表现越好；如果权重持续偏低，说明当前任务对简单树桩较困难。",
            "需要警惕异常值。AdaBoost 会持续提高难分样本权重，如果这些样本本身是噪声，模型可能会把大量迭代消耗在少数异常点上，导致泛化表现下降。",
        ),
    )


def _run_mlp(data: PreparedData) -> dict[str, Any]:
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.01,
        max_iter=350,
        random_state=42,
    )
    model.fit(data.x_train, data.y_train)
    predictions = model.predict(data.x_test)
    accuracy = accuracy_score(data.y_test, predictions)
    loss_curve = [
        {"epoch": index + 1, "loss": round(float(loss), 4)}
        for index, loss in enumerate(model.loss_curve_)
    ]
    topology = [
        {"layer": "输入层", "neurons": int(data.x_train.shape[1])},
        {"layer": "隐藏层 1", "neurons": 16},
        {"layer": "隐藏层 2", "neurons": 8},
        {"layer": "输出层", "neurons": len(data.labels)},
    ]
    return _standard_payload(
        data,
        model,
        predictions,
        accuracy,
        charts={
            "type": "mlp",
            "scatter": _scatter(data.x_all_2d, data.y_all, data.labels),
            "lossCurve": loss_curve,
            "topology": topology,
            "canvas": {
                "mode": "network",
                **_split_canvas(data, model),
                "topology": topology,
                "steps": [
                    {"title": "展示训练集与测试集", "detail": "训练集用于反向传播训练，测试集用于最终验证。"},
                    {"title": "训练神经网络", "detail": "MLP 在完整特征空间中学习非线性映射。"},
                    {"title": "预测测试集", "detail": "测试集经过输出层得到预测类别。"},
                    {"title": "对照真实标签", "detail": "测试集按预测类别着色，错误预测用橙色描边标出。"},
                ],
            },
        },
        analysis=_analysis(
            "MLP 的优势在于表达非线性关系。相比逻辑回归，它可以通过隐藏层组合特征，让分类边界出现弯曲、折叠和局部变化。",
            "训练 loss 曲线是判断优化过程的重要依据。若 loss 持续下降并逐渐平稳，说明模型正在收敛；若曲线震荡明显，可能需要降低学习率或增加正则化。",
            "MLP 的结果不能只看最终 Accuracy。训练准确率、测试准确率和泛化差距能揭示是否过拟合；Macro F1 能帮助判断模型是否兼顾了所有类别。",
            "教学分析中应强调 MLP 的代价：它更依赖超参数和数据预处理，可解释性也弱于决策树。适合展示复杂边界，但不应默认替代所有传统模型。",
        ),
    )


def _standard_payload(
    data: PreparedData,
    model: Any,
    predictions: np.ndarray,
    accuracy: float,
    charts: dict[str, Any],
    analysis: str,
) -> dict[str, Any]:
    matrix = confusion_matrix(data.y_test, predictions).tolist()
    train_predictions = model.predict(data.x_train)
    train_accuracy = accuracy_score(data.y_train, train_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        data.y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        data.y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        data.y_test,
        predictions,
        labels=list(range(len(data.labels))),
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(data.y_test, predictions)
    macro_recall = np.mean(
        [
            matrix[index][index] / max(1, sum(matrix[index]))
            for index in range(len(matrix))
        ]
    )
    return {
        "charts": {
            **charts,
            "confusionMatrix": {
                "labels": data.labels,
                "values": matrix,
            },
            "classMetrics": [
                {
                    "label": label,
                    "precision": round(float(class_precision[index]), 4),
                    "recall": round(float(class_recall[index]), 4),
                    "f1": round(float(class_f1[index]), 4),
                    "support": int(class_support[index]),
                }
                for index, label in enumerate(data.labels)
            ],
        },
        "metrics": [
            {"label": "Accuracy", "value": round(float(accuracy), 4), "format": "percent"},
            {"label": "Train Accuracy", "value": round(float(train_accuracy), 4), "format": "percent"},
            {"label": "Generalization Gap", "value": round(float(train_accuracy - accuracy), 4), "format": "percent"},
            {"label": "Balanced Accuracy", "value": round(float(balanced_accuracy), 4), "format": "percent"},
            {"label": "Macro Precision", "value": round(float(precision), 4), "format": "percent"},
            {"label": "Macro Recall", "value": round(float(macro_recall), 4), "format": "percent"},
            {"label": "Macro F1", "value": round(float(f1), 4), "format": "percent"},
            {"label": "Weighted Precision", "value": round(float(weighted_precision), 4), "format": "percent"},
            {"label": "Weighted Recall", "value": round(float(weighted_recall), 4), "format": "percent"},
            {"label": "Weighted F1", "value": round(float(weighted_f1), 4), "format": "percent"},
            {"label": "训练样本", "value": int(len(data.y_train)), "format": "integer"},
            {"label": "测试样本", "value": int(len(data.y_test)), "format": "integer"},
        ],
        "analysis": analysis,
    }


def _scatter(points: np.ndarray, labels: np.ndarray, label_names: list[str]):
    return [
        _point(point, int(label), label_names)
        for point, label in zip(points, labels)
    ]


def _predicted_canvas_points(
    points: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    model: Any,
):
    model.fit(points, labels)
    predictions = model.predict(points)
    canvas_points = []
    for point, actual, predicted in zip(points, labels, predictions):
        item = _point(point, int(actual), label_names)
        predicted_index = int(predicted)
        item["actualLabel"] = label_names[int(actual)]
        item["actualLabelIndex"] = int(actual)
        item["predictedLabel"] = label_names[predicted_index]
        item["predictedLabelIndex"] = predicted_index
        item["predictedColor"] = PALETTE[predicted_index % len(PALETTE)]
        item["isCorrect"] = bool(int(actual) == predicted_index)
        return_color = "#111827" if item["isCorrect"] else "#f59e0b"
        item["resultStroke"] = return_color
        canvas_points.append(item)
    return canvas_points


def _split_canvas(data: PreparedData, model: Any):
    train_predictions = model.predict(data.x_train)
    test_predictions = model.predict(data.x_test)
    return {
        "train": _canvas_points(data.x_train_2d, data.y_train, train_predictions, data.labels, "train"),
        "test": _canvas_points(data.x_test_2d, data.y_test, test_predictions, data.labels, "test"),
    }


def _lda_canvas(data: PreparedData, model: LinearDiscriminantAnalysis):
    train_projection = model.transform(data.x_train)
    test_projection = model.transform(data.x_test)
    train_projection = _ensure_2d_projection(train_projection)
    test_projection = _ensure_2d_projection(test_projection)
    train_predictions = model.predict(data.x_train)
    test_predictions = model.predict(data.x_test)
    return {
        "train": _canvas_points(data.x_train_2d, data.y_train, train_predictions, data.labels, "train"),
        "test": _canvas_points(data.x_test_2d, data.y_test, test_predictions, data.labels, "test"),
        "projectedTrain": _canvas_points(train_projection, data.y_train, train_predictions, data.labels, "train"),
        "projectedTest": _canvas_points(test_projection, data.y_test, test_predictions, data.labels, "test"),
    }


def _knn_canvas(data: PreparedData, model: KNeighborsClassifier):
    test_predictions = model.predict(data.x_test)
    test_points = _canvas_points(data.x_test_2d, data.y_test, test_predictions, data.labels, "test")
    train_points = _canvas_points(data.x_train_2d, data.y_train, data.y_train, data.labels, "train")
    distances, indices = model.kneighbors(data.x_test, n_neighbors=min(25, len(data.x_train)))
    neighbor_sets = []
    for test_index, test_point in enumerate(test_points):
        neighbor_sets.append(
            {
                "testIndex": test_index,
                "testPoint": test_point,
                "neighbors": [
                    {
                        "point": train_points[int(train_index)],
                        "distance": round(float(distances[test_index][rank]), 4),
                        "rank": rank + 1,
                    }
                    for rank, train_index in enumerate(indices[test_index])
                ],
            }
        )
    return {
        "train": train_points,
        "test": test_points,
        "neighborSets": neighbor_sets,
    }


def _canvas_points(
    points: np.ndarray,
    actual_labels: np.ndarray,
    predicted_labels: np.ndarray,
    label_names: list[str],
    split: str,
):
    result = []
    for index, (point, actual, predicted) in enumerate(zip(points, actual_labels, predicted_labels)):
        actual_index = int(actual)
        predicted_index = int(predicted)
        item = _point(point, actual_index, label_names)
        item["id"] = f"{split}-{index}"
        item["split"] = split
        item["actualLabel"] = label_names[actual_index]
        item["actualLabelIndex"] = actual_index
        item["predictedLabel"] = label_names[predicted_index]
        item["predictedLabelIndex"] = predicted_index
        item["predictedColor"] = PALETTE[predicted_index % len(PALETTE)]
        item["isCorrect"] = bool(actual_index == predicted_index)
        result.append(item)
    return result


def _ensure_2d_projection(values: np.ndarray):
    if values.shape[1] == 1:
        return np.column_stack([values[:, 0], np.zeros(values.shape[0])])
    return values[:, :2]


def _point(point: np.ndarray, label: int, label_names: list[str]):
    return {
        "x": round(float(point[0]), 4),
        "y": round(float(point[1]), 4),
        "label": label_names[label],
        "labelIndex": label,
        "color": PALETTE[label % len(PALETTE)],
    }


def _feature_importance(model: DecisionTreeClassifier, feature_names: list[str]):
    values = sorted(
        [
            {"feature": feature, "importance": round(float(importance), 4)}
            for feature, importance in zip(feature_names, model.feature_importances_)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return values[:8]


def _tree_nodes(model: DecisionTreeClassifier, feature_names: list[str], label_names: list[str]):
    tree = model.tree_
    nodes = []
    for index in range(min(tree.node_count, 31)):
        is_leaf = tree.children_left[index] == tree.children_right[index]
        values = tree.value[index][0]
        predicted = int(np.argmax(values))
        nodes.append(
            {
                "id": int(index),
                "left": int(tree.children_left[index]) if not is_leaf else None,
                "right": int(tree.children_right[index]) if not is_leaf else None,
                "feature": None if is_leaf else feature_names[int(tree.feature[index])],
                "threshold": None if is_leaf else round(float(tree.threshold[index]), 3),
                "samples": int(tree.n_node_samples[index]),
                "impurity": round(float(tree.impurity[index]), 4),
                "prediction": label_names[predicted],
                "color": PALETTE[predicted % len(PALETTE)],
                "leaf": bool(is_leaf),
            }
        )
    return nodes


def _boosting_frames(points: np.ndarray, labels: np.ndarray, label_names: list[str]):
    weights = np.ones(len(labels)) / len(labels)
    frames = []
    for round_index in range(6):
        focus = np.abs(points[:, 0] - np.median(points[:, 0])) + np.abs(points[:, 1] - np.median(points[:, 1]))
        focus = focus / max(float(focus.max()), 1e-6)
        weights = weights * (1 + focus * (0.18 + round_index * 0.03))
        weights = weights / weights.sum()
        frames.append(
            {
                "round": round_index + 1,
                "points": [
                    {
                        **_point(point, int(label), label_names),
                        "weight": round(float(weight), 5),
                    }
                    for point, label, weight in zip(points, labels, weights)
                ],
            }
        )
    return frames


def _analysis(*paragraphs: str):
    titles = ["总体判断", "可视化解读", "指标解读", "教学建议"]
    return [
        {"title": title, "body": body}
        for title, body in zip(titles, paragraphs)
    ]
