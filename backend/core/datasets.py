from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_wine


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"


DATASETS = {
    "iris": {
        "id": "iris",
        "name": "Iris 鸢尾花数据集",
        "file": "iris.csv",
        "loader": load_iris,
        "source": "UCI Machine Learning Repository / scikit-learn built-in datasets",
        "description": "经典多分类数据集，包含 150 条鸢尾花样本与 4 个形态特征，用于区分 Setosa、Versicolor、Virginica。",
        "task": "三分类",
        "targetLabel": "species",
    },
    "wine": {
        "id": "wine",
        "name": "Wine 葡萄酒数据集",
        "file": "wine.csv",
        "loader": load_wine,
        "source": "UCI Machine Learning Repository / scikit-learn built-in datasets",
        "description": "经典化学成分分类数据集，包含 178 条样本与 13 个特征，用于根据葡萄酒化学指标识别产区类别。",
        "task": "三分类",
        "targetLabel": "class",
    },
}


def load_dataset(dataset_id: str):
    config = DATASETS[dataset_id]
    raw = config["loader"]()
    frame = pd.DataFrame(raw.data, columns=list(raw.feature_names))
    target_names = [str(name) for name in raw.target_names]
    frame[config["targetLabel"]] = [target_names[int(index)] for index in raw.target]
    return {
        "config": config,
        "raw": raw,
        "frame": frame,
        "featureNames": list(raw.feature_names),
        "targetNames": target_names,
        "target": raw.target,
        "data": raw.data,
    }


def ensure_dataset_files():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id, config in DATASETS.items():
        path = DATASET_DIR / config["file"]
        if not path.exists():
            loaded = load_dataset(dataset_id)
            loaded["frame"].to_csv(path, index=False, encoding="utf-8")


def get_dataset_summary(dataset_id: str, include_preview: bool = False):
    loaded = load_dataset(dataset_id)
    frame = loaded["frame"]
    feature_names = loaded["featureNames"]
    target_label = loaded["config"]["targetLabel"]
    target_counts = frame[target_label].value_counts().to_dict()
    numeric_description = frame[feature_names].describe().round(3)

    summary = {
        "id": loaded["config"]["id"],
        "name": loaded["config"]["name"],
        "file": str((DATASET_DIR / loaded["config"]["file"]).relative_to(ROOT_DIR)),
        "source": loaded["config"]["source"],
        "description": loaded["config"]["description"],
        "task": loaded["config"]["task"],
        "sampleCount": int(frame.shape[0]),
        "featureCount": len(feature_names),
        "targetLabel": target_label,
        "targetNames": loaded["targetNames"],
        "targetDistribution": [
            {"label": str(label), "count": int(count)}
            for label, count in target_counts.items()
        ],
        "fields": [
            {
                "name": name,
                "type": "number",
                "mean": float(numeric_description.loc["mean", name]),
                "std": float(numeric_description.loc["std", name]),
                "min": float(numeric_description.loc["min", name]),
                "max": float(numeric_description.loc["max", name]),
            }
            for name in feature_names
        ],
    }

    if include_preview:
        summary["preview"] = frame.head(12).round(3).to_dict(orient="records")

    return summary
