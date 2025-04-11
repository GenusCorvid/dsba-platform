"""
This module provides functionality for model selection and performance tracking.
It allows comparing different models and selecting the best one based on configurable criteria.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import pandas as pd
from sklearn.base import BaseEstimator
import logging
from dsba.model_evaluation import evaluate_classifier, ClassifierEvaluationResult
from dsba.model_registry import ClassifierMetadata, save_model
from datetime import datetime

@dataclass
class ModelCandidate:
    """A data class to store information about a model candidate."""
    model: BaseEstimator
    name: str
    hyperparameters: Dict[str, Any]
    evaluation_result: Optional[ClassifierEvaluationResult] = None
    
    def evaluate(self, data: pd.DataFrame, target_column: str) -> ClassifierEvaluationResult:
        """Evaluate the model on the provided data and store the evaluation result."""
        self.evaluation_result = evaluate_classifier(self.model, target_column, data)
        return self.evaluation_result


def select_best_model(
    candidates: List[ModelCandidate], 
    validation_data: pd.DataFrame, 
    target_column: str,
    metric: str = "f1_score",
    greater_is_better: bool = True
) -> ModelCandidate:
    """
    Select the best model from a list of candidates based on a specified metric.
    
    Args:
        candidates: List of ModelCandidate objects
        validation_data: DataFrame to use for validation
        target_column: Name of the target column
        metric: Metric to use for selection ('accuracy', 'precision', 'recall', or 'f1_score')
        greater_is_better: Whether higher values of the metric are better
        
    Returns:
        The best model candidate
    """
    if metric not in ["accuracy", "precision", "recall", "f1_score"]:
        raise ValueError(f"Unsupported metric: {metric}. Must be one of 'accuracy', 'precision', 'recall', or 'f1_score'")
    
    # Evaluate all candidates if they haven't been evaluated yet
    for candidate in candidates:
        if candidate.evaluation_result is None:
            candidate.evaluate(validation_data, target_column)
    
    # Select the best model based on the specified metric
    if greater_is_better:
        best_candidate = max(candidates, key=lambda c: getattr(c.evaluation_result, metric))
    else:
        best_candidate = min(candidates, key=lambda c: getattr(c.evaluation_result, metric))
        
    logging.info(f"Selected {best_candidate.name} as the best model based on {metric}")
    return best_candidate


def save_best_model(
    best_candidate: ModelCandidate,
    model_id: str,
    target_column: str,
    algorithm: str,
    description: str = ""
) -> ClassifierMetadata:
    """
    Save the best model to the model registry with its performance metrics.
    
    Args:
        best_candidate: The best model candidate
        model_id: ID to use for the model
        target_column: Name of the target column
        algorithm: Name of the algorithm
        description: Optional description of the model
        
    Returns:
        The metadata of the saved model
    """
    if best_candidate.evaluation_result is None:
        raise ValueError("Model must be evaluated before saving")
    
    # Create metadata with performance metrics
    performance_metrics = {
        "accuracy": best_candidate.evaluation_result.accuracy,
        "precision": best_candidate.evaluation_result.precision,
        "recall": best_candidate.evaluation_result.recall,
        "f1_score": best_candidate.evaluation_result.f1_score
    }
    
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm=algorithm,
        target_column=target_column,
        hyperparameters=best_candidate.hyperparameters,
        description=description,
        performance_metrics=performance_metrics
    )
    
    # Save the model with metadata
    save_model(best_candidate.model, metadata)
    logging.info(f"Saved best model with ID: {model_id}")
    
    return metadata


def compare_models(
    candidates: List[ModelCandidate],
    validation_data: pd.DataFrame, 
    target_column: str
) -> pd.DataFrame:
    """
    Compare multiple models and return a DataFrame with their performance metrics.
    
    Args:
        candidates: List of ModelCandidate objects
        validation_data: DataFrame to use for validation
        target_column: Name of the target column
        
    Returns:
        DataFrame with performance metrics for each model
    """
    # Evaluate all candidates if they haven't been evaluated yet
    for candidate in candidates:
        if candidate.evaluation_result is None:
            candidate.evaluate(validation_data, target_column)
    
    # Create a DataFrame with performance metrics
    comparison_data = []
    for candidate in candidates:
        result = candidate.evaluation_result
        comparison_data.append({
            "Model": candidate.name,
            "Accuracy": result.accuracy,
            "Precision": result.precision,
            "Recall": result.recall,
            "F1 Score": result.f1_score
        })
    
    return pd.DataFrame(comparison_data)
