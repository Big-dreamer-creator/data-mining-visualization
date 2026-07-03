from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.frame_algorithms import ALGORITHMS, run_algorithm
from core.datasets import DATASETS, ensure_dataset_files, get_dataset_summary
from core.evaluation import evaluate_algorithm

router = APIRouter()


class AlgorithmRunRequest(BaseModel):
    pattern: str = "blobs"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class AlgorithmEvaluationRequest(BaseModel):
    datasetId: str = "iris"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/algorithms")
def list_algorithms():
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "shortName": item["short_name"],
            "category": item["category"],
            "summary": item["summary"],
        }
        for item in ALGORITHMS.values()
    ]


@router.get("/algorithms/{algorithm_id}")
def get_algorithm(algorithm_id: str):
    algorithm = ALGORITHMS.get(algorithm_id)
    if algorithm is None:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return algorithm


@router.get("/datasets")
def list_datasets():
    ensure_dataset_files()
    return [get_dataset_summary(dataset_id) for dataset_id in DATASETS]


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_files()
    return get_dataset_summary(dataset_id, include_preview=True)


@router.post("/algorithms/{algorithm_id}/visualization")
def run_visualization_endpoint(algorithm_id: str, payload: AlgorithmRunRequest):
    if algorithm_id not in ALGORITHMS:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    try:
        return run_algorithm(
            algorithm_id,
            pattern=payload.pattern,
            hyperparameters=payload.hyperparameters,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.post("/algorithms/{algorithm_id}/evaluate")
def evaluate_algorithm_endpoint(algorithm_id: str, payload: AlgorithmEvaluationRequest):
    if algorithm_id not in ALGORITHMS:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    if payload.datasetId not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        return evaluate_algorithm(
            algorithm_id,
            dataset_id=payload.datasetId,
            hyperparameters=payload.hyperparameters,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.post("/algorithms/{algorithm_id}/run")
def run_algorithm_endpoint(algorithm_id: str, payload: AlgorithmRunRequest):
    return run_visualization_endpoint(algorithm_id, payload)
