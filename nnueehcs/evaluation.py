import torch
import torch.nn as nn
from typing import Union, Tuple, Callable
import numpy as np
import time
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from abc import ABC, abstractmethod
from .classification import PercentileBasedIdOodClassifier, ReversedPercentileBasedIdOodClassifier
from .utility import ResultsInstance


class UncertaintyEstimate:
    def __init__(self, data: Union[np.ndarray, Tuple]):
        """Initialize UncertaintyEstimate with data
        
        Args:
            data: Input data as numpy array, torch tensor, or tuple of these
            
        Raises:
            ValueError: If data is empty or has invalid shape/values
        """
        if isinstance(data, (np.ndarray, torch.Tensor)) and data.size == 0:
            raise ValueError("Cannot create UncertaintyEstimate from empty data")
        elif isinstance(data, tuple) and any(d.size == 0 for d in data):
            raise ValueError("Cannot create UncertaintyEstimate from empty tuple data")
            
        self.data = self._to_numpy(data)
        
        # Validate tuple data has matching dimensions
        if isinstance(self.data, tuple):
            shapes = [d.shape[0] for d in self.data]
            if len(set(shapes)) > 1:
                raise ValueError(f"All arrays in tuple must have same first dimension, got shapes: {shapes}")

    @property
    def dimensions(self) -> int:
        return len(self.data) if isinstance(self.data, tuple) else 1

    def flatten(self):
        if self.dimensions != 1:
            raise ValueError("Can only flatten 1D uncertainty estimates")
        return self.data.flatten()

    def mean(self):
        """Calculate mean of uncertainty estimate
        
        Returns:
            float: Mean value across all dimensions
            
        Note:
            Returns NaN if data contains NaN values
        """
        return np.mean(self._combine())

    def _combine(self):
        """Combine multi-dimensional data into single array
        
        Returns:
            np.ndarray: Combined data
            
        Raises:
            ValueError: If dimensions don't match for tuple data
        """
        if self.dimensions == 1:
            return self.data
        else:
            try:
                flat_dat = [d.flatten() for d in self.data]
                return np.concatenate(flat_dat)
            except ValueError as e:
                raise ValueError(f"Failed to combine data dimensions: {e}")

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, Tuple]) -> Union[np.ndarray, Tuple]:
        """Convert input data to numpy array(s)
        
        Args:
            data: Input data to convert
            
        Returns:
            Converted numpy array or tuple of arrays
            
        Raises:
            TypeError: If input data type is not supported
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, tuple):
            return tuple(self._to_numpy(d) for d in data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")


class EvaluationMetric(ABC):
    """Base class for all evaluation metrics (both uncertainty and classification)"""
    @abstractmethod
    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        pass

    @classmethod
    @abstractmethod
    def get_objectives(cls):
        """Return list of objectives for optimization during training"""
        pass

    @classmethod
    @abstractmethod
    def get_metrics(cls):
        """Return list of all metrics this evaluator can compute"""
        pass

    @abstractmethod
    def get_name(cls):
        """Return name of the metric"""
        pass


class TrainingMetric(ABC):
    """Base class for training metrics"""
    @abstractmethod
    def evaluate(self, results_instance: ResultsInstance) -> dict:
        pass
    @classmethod
    @abstractmethod
    def get_name(cls):
        """Return name of the metric"""
        pass
    @classmethod
    @abstractmethod
    def get_objectives(cls):
        """Return list of objectives for optimization during training"""
        pass

    @classmethod
    @abstractmethod
    def get_metrics(cls):
        """Return list of all metrics this evaluator can compute"""
        pass

class TrainingTimeMetric(TrainingMetric):
    name = "training_time"
    """Base class for training time metrics"""
    def evaluate(self, results_instance: ResultsInstance) -> dict:
        trial_num = results_instance.get_trial_number()
        results_file = results_instance.get_trial_results_file()
        results = pd.read_csv(results_file)
        return {self.name: results.iloc[trial_num]['train_time']}
    
    @classmethod
    def get_name(cls):
        return cls.name

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "minimize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]

class UncertaintyEvaluationMetric(EvaluationMetric):
    """Base class for uncertainty evaluation metrics"""
    
    def evaluate(self, model, id_data: tuple, ood_data: tuple) -> dict:
        """Evaluate uncertainty estimates from model predictions
        
        Args:
            model: Model that returns uncertainty estimates
            id_data: Tuple of (inputs, outputs) for in-distribution data
            ood_data: Tuple of (inputs, outputs) for out-of-distribution data
            
        Returns:
            Dictionary containing evaluation metric(s)
        """
        model.eval()
        with torch.no_grad():
            _, id_scores = model(id_data[0], return_ue=True)
            _, ood_scores = model(ood_data[0], return_ue=True)
            
        id_ue = UncertaintyEstimate(id_scores)
        ood_ue = UncertaintyEstimate(ood_scores)
        
        result = self._evaluate_uncertainties(id_ue, ood_ue)
        
        # Ensure all values are Python floats
        return {k: float(v) for k, v in result.items()}

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        """Implement specific evaluation metric
        
        Args:
            id_ue: UncertaintyEstimate for in-distribution data
            ood_ue: UncertaintyEstimate for out-of-distribution data
            
        Returns:
            Dictionary containing evaluation metric(s)
        """
        raise NotImplementedError


class ClassificationMetric(EvaluationMetric):
    """Base class for classification-based metrics like TNR@TPR95, AUROC, etc."""
    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        with torch.no_grad():
            _, id_scores = model(id_data[0], return_ue=True)
            _, ood_scores = model(ood_data[0], return_ue=True)
        return self._evaluate_scores(id_scores, ood_scores)

    @abstractmethod
    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        pass


class WassersteinEvaluation(UncertaintyEvaluationMetric):
    name = "wasserstein_distance"

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        value = None

        if id_ue.dimensions == 1:
            value = wasserstein_distance(id_ue.flatten(), ood_ue.flatten())
        else:
            distances = [wasserstein_distance(id_ue.data[i].flatten(), 
                                              ood_ue.data[i].flatten()) 
                         for i in range(id_ue.dimensions)]
            value = np.mean(distances)
        return {self.name: value}

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]

    def get_name(self):
        return self.name


class EuclideanEvaluation(UncertaintyEvaluationMetric):
    name = "euclidean_distance"

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        result = np.mean(np.sqrt(np.sum((id_ue.data - ood_ue.data) ** 2, axis=-1)))
        return {self.name: float(result)}

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]

    def get_name(self):
        return self.name


class JensenShannonEvaluation(UncertaintyEvaluationMetric):
    name = "jensen_shannon_distance"

    def _to_probability_distribution(self, ue: UncertaintyEstimate) -> np.ndarray:
        if ue.dimensions == 1:
            return ue.data / np.sum(ue.data)
        else:
            return np.array([d / np.sum(d) for d in ue.data])

    def _is_probability_distribution(self, data: np.ndarray) -> bool:
        return np.allclose(np.sum(data), 1.0)

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")

        p1 = id_ue.data
        p2 = ood_ue.data

        result = self._average_js_distance(p1, p2)
        return {self.name: result}

    def _average_js_distance(self, array1: np.array, array2: np.array) -> float:
        from scipy.spatial.distance import jensenshannon
        p1 = array1
        p2 = array2

        if p1.ndim == 1 or (p1.ndim == 2 and p1.shape[1] == 1):
            p1flat = p1.flatten()
            p2flat = p2.flatten()
            # extend with zeros so their shapes match
            # js_distances = jensenshannon(p1flat, p2flat)
            return self.pdf_jsd(p1flat, p2flat)
        else:
            js_distances = [jensenshannon(p1[i], p2[i]) for i in range(p1.shape[0])]

        return np.mean(js_distances)

    def pdf_jsd(self, dist1, dist2, num_points=20000):
        from scipy.stats import gaussian_kde
        from scipy.spatial.distance import jensenshannon
        kde1 = gaussian_kde(dist1)
        kde2 = gaussian_kde(dist2)
        x_range = np.linspace(min(dist1.min(), dist2.min()), max(dist1.max(), dist2.max()), num_points)
        pdf1 = kde1(x_range)
        pdf2 = kde2(x_range)
        return jensenshannon(pdf1, pdf2)

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name]

    def get_name(self):
        return self.name

class MeanScoreEvaluation(UncertaintyEvaluationMetric):
    """Evaluates the mean uncertainty score of the model.
       This is intended to be used as a minimization target for Bayesian Optimization.
       We want to parameterize the UE technique such that UE scores of ID data are minimized.
       This, hopefully, gives good downstream metrics without required OOD data at training time.
    """
    name = "mean_score"

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")
        result = np.mean(id_ue.data)
        return {self.name: result}
    
    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "minimize"
        }]
    
    @classmethod
    def get_metrics(cls):
        return [cls.name]
    
    def get_name(self):
        return self.name

class MaxScoreEvaluation(UncertaintyEvaluationMetric):
    name = "max_score"

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        result = np.max(id_ue.data)
        return {self.name: result}

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]
    
    @classmethod
    def get_metrics(cls):
        return [cls.name]
    
    def get_name(self):
        return self.name

class PercentileScoreEvaluation(UncertaintyEvaluationMetric):
    """Evaluates a specific percentile of uncertainty scores.
       This allows for evaluating different thresholds (e.g., 90th, 95th percentile)
       without using the maximum score, which might be sensitive to outliers.
    """
    name = "percentile_score"
    
    def __init__(self, percentile: float = 95.0):
        """
        Args:
            percentile: The percentile to evaluate (between 0 and 100)
        """
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
        self.percentile = percentile
    
    @classmethod
    def from_config(cls, config: dict) -> 'PercentileScoreEvaluation':
        """Factory method to create from config dictionary"""
        print(config)
        return cls(percentile=config.get('percentile', 95.0))

    def _evaluate_uncertainties(self, id_ue: UncertaintyEstimate, ood_ue: UncertaintyEstimate) -> dict:
        if id_ue.dimensions != ood_ue.dimensions:
            raise ValueError("Uncertainty estimates must have the same dimensions")
        result = np.percentile(id_ue.data, self.percentile)
        return {self.name: result}
    
    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "minimize"
        }]
    
    @classmethod
    def get_metrics(cls):
        return [cls.name]
    
    def get_name(self):
        return self.name

class MaxMemoryUsageEvaluation(EvaluationMetric):
    name = "max_memory_usage"

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        import gc

        model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            id_ood_combined = torch.cat([id_data[0], ood_data[0]])
            _, _ = model(id_ood_combined, return_ue=True)
            max_memory_usage = torch.cuda.max_memory_allocated()
            max_memory_usage_mb = max_memory_usage / (1024 * 1024)
        return {
            'max_memory_usage': max_memory_usage_mb
        }

    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "minimize"
        }]

    def get_metrics(cls):
        return [cls.name]

    def get_name(self):
        return self.name

class RuntimeEvaluation(EvaluationMetric):
    name = "runtime"
    def __init__(self, num_trials: int = 20, num_warmup: int = 5):
        self.num_trials = num_trials
        self.num_warmup = num_warmup

    @classmethod
    def from_config(cls, config: dict) -> 'RuntimeEvaluation':
        """Factory method to create from config dictionary"""
        return cls(
            num_trials=config.get('trials', 20),
            num_warmup=config.get('warmup', 5)
        )

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        raise NotImplementedError("Cannot call evaluate on base class")

    def _evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple, eval_functor: Callable, return_raw: bool = False) -> dict:
        model.eval()
        runtimes = np.zeros(self.num_trials)
        data_combined = torch.cat([id_data[0], ood_data[0]])
        with torch.no_grad():
            for _ in range(self.num_warmup):
                eval_functor(model, data_combined)
            for trial in range(self.num_trials):
                start_time = time.time()
                retval = eval_functor(model, data_combined)
                torch.cuda.synchronize()
                end_time = time.time()
                runtimes[trial] = end_time - start_time
        mean = np.mean(runtimes)
        std = np.std(runtimes)
        if return_raw:
            return {'runtime': mean, 'runtime_std': std, 'runtimes': runtimes}
        else:
            return {'runtime': mean, 'runtime_std': std}

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "minimize"
        }]

    @classmethod
    def get_metrics(cls):
        return [cls.name, 'runtime_std']

    def get_name(self):
        return self.name

class BaseModelRuntimeEvaluation(RuntimeEvaluation):
    name = "base_model_runtime"

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        callable = lambda model, data: model(data)
        return super()._evaluate(model, id_data, ood_data, callable)

class UncertaintyEstimatingRuntimeEvaluation(RuntimeEvaluation):
    name = "uncertainty_estimating_runtime"

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        callable = lambda model, data: model(data, return_ue=True)
        return super()._evaluate(model, id_data, ood_data, callable)

class BaseModelThroughputEvaluation(RuntimeEvaluation):
    name = "base_model_throughput"

    def _convert_to_throughput(self, runtimes: dict, total_samples: int) -> float:
        runtimes = runtimes['runtimes']
        throughput = total_samples / runtimes
        throughput_mean = np.mean(throughput)
        throughput_std = np.std(throughput)
        return throughput_mean, throughput_std

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        runtimes = super()._evaluate(model, id_data, ood_data, lambda model, data: model(data), return_raw=True)
        total_samples = id_data[0].shape[0] + ood_data[0].shape[0]
        throughput_mean, throughput_std = self._convert_to_throughput(runtimes, total_samples)
        return {self.name: throughput_mean, 'throughput_std': throughput_std}

class UncertaintyEstimatingThroughputEvaluation(BaseModelThroughputEvaluation):
    name = "uncertainty_estimating_throughput"
    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        runtimes = super()._evaluate(model, id_data, ood_data, lambda model, data: model(data, return_ue=True), return_raw=True)
        total_samples = id_data[0].shape[0] + ood_data[0].shape[0]
        throughput_mean, throughput_std = self._convert_to_throughput(runtimes, total_samples)
        return {self.name: throughput_mean, 'throughput_std': throughput_std}

    @classmethod
    def get_objectives(cls):
        return [{
            "name": cls.name,
            "type": "maximize"
        }]
        
    @classmethod
    def get_metrics(cls):
        return [cls.name]

    @classmethod
    def get_name(cls):
        return cls.name


class TNRatTPX(ClassificationMetric):
    """Calculates True Negative Rate (TNR) at a specified True Positive Rate (TPR)"""
    def __init__(self, target_tpr: float, reversed: bool = False):
        """
        Args:
            target_tpr: The TPR level at which to calculate TNR (between 0 and 1)
        """
        if not 0 <= target_tpr <= 1:
            raise ValueError(f"target_tpr must be between 0 and 1, got {target_tpr}")
        self.target_tpr = target_tpr
        # Create metric name based on percentage (e.g., 'tnr_at_tpr95' for 0.95)
        self.metric_name = f'tnr_at_tpr'
        self.reversed = reversed

    @classmethod
    def from_config(cls, config: dict) -> 'TNRatTPX':
        """Factory method to create from config dictionary"""
        return cls(target_tpr=config['target_tpr'], reversed=config.get('reversed', False))

    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        # Flatten scores
        id_scores = id_scores.reshape(-1)
        ood_scores = ood_scores.reshape(-1)
    
        # Perfect separation check
        if self.reversed:
            min_id = id_scores.min()
            max_ood = ood_scores.max()
            if min_id > max_ood:
                return {str(self): 1.0}
        else:
            max_id = id_scores.max()
            min_ood = ood_scores.min()
            if max_id < min_ood:
                return {str(self): 1.0}
    
        # Sort all scores together to get thresholds
        all_scores = torch.cat([id_scores, ood_scores])
        thresholds = torch.unique(all_scores)
    
        # Arrays to store results
        best_tnr = 0.0
    
        n_id = len(id_scores)
        n_ood = len(ood_scores)
    
        for threshold in thresholds:
            if self.reversed:
                tp = (id_scores > threshold).sum().item()
                tn = (ood_scores <= threshold).sum().item()
            else:
                tp = (ood_scores > threshold).sum().item()
                tn = (id_scores <= threshold).sum().item()

            tpr = tp / n_ood if n_ood > 0 else 0
            tnr = tn / n_id if n_id > 0 else 0

            # If we found a TPR that exceeds our target, update best TNR
            if tpr >= self.target_tpr and tnr > best_tnr:
                best_tnr = tnr

        return {str(self): best_tnr}

    @classmethod
    def get_objectives(cls):
        # Note: This is a class method, so we can't access self.metric_name
        # Instead, the actual metric name will be set when instantiating
        return [{'name': 'tnr_at_tpr', 'type': 'maximize'}]

    @classmethod
    def get_metrics(cls):
        # Similar to get_objectives, actual name set during instantiation
        return ['tnr_at_tpr']

    def get_instance_objectives(self):
        """Instance-specific objectives with correct metric name"""
        return [{'name': self.metric_name, 'type': 'maximize'}]

    def get_instance_metrics(self):
        """Instance-specific metrics with correct metric name"""
        return [self.metric_name]

    def get_name(self):
        return f'{self.metric_name}{int(100*self.target_tpr)}'

    def __str__(self):
        return self.get_name()

class AUROC(ClassificationMetric):
    """
    Calculate AUROC given some percentile-based classification
    threshold.
    """
    name = "auroc"

    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        # Convert to numpy arrays and flatten
        id_scores = id_scores.cpu().numpy().flatten()
        ood_scores = ood_scores.cpu().numpy().flatten()

        # Combine scores and create true labels (1 for OOD, 0 for ID)
        y_scores = np.concatenate([id_scores, ood_scores])
        y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])

        # Compute AUROC using raw scores
        return {self.name: roc_auc_score(y_true, y_scores)}

    @classmethod
    def get_objectives(cls):
        return [{'name': 'auroc', 'type': 'maximize'}]

    @classmethod
    def get_metrics(cls):
        return ['auroc']
    
    def get_name(self):
        return self.name

class PercentileBasedClassifier(ClassificationMetric):
    def __init__(self, percentile: float, reversed: bool = False):
        self._classifier = PercentileBasedIdOodClassifier(percentile)
        self.reversed = reversed

    def _evaluate_scores(self, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
        if self.reversed:
            results = self._classifier._evaluate_scores(-id_scores, -ood_scores)
        else:
            results = self._classifier._evaluate_scores(id_scores, ood_scores)
        return {k: v for k, v in results.items() if k in self.get_metrics()}

    @classmethod
    def get_objectives(cls):
        return [{'name': 'sensitivity', 'type': 'maximize'},
                {'name': 'specificity', 'type': 'maximize'}]

    @classmethod
    def get_metrics(cls):
        return ['sensitivity', 'specificity']

    def get_name(self):
        suffix = f'_{int(100*self._classifier.percentile)}'
        if self.reversed:
            suffix = f'_reversed{suffix}'
        return f'percentile_classification{suffix}'



class MetricEvaluator:
    """Unified evaluator that can handle multiple metrics"""
    def __init__(self, metrics: list[EvaluationMetric]):
        self.metrics = metrics

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple, results_instance=None) -> dict:
        results = {}
        for metric in self.metrics:
            if is_training_metric(metric):
                if results_instance is None:
                    raise ValueError(f"Training metric {metric.get_name()} requires results_instance")
                results.update(metric.evaluate(results_instance))
            else:
                results.update(metric.evaluate(model, id_data, ood_data))
        return results

    def get_training_objectives(self):
        """Get objectives for optimization during training"""
        objectives = []
        for metric in self.metrics:
            # Use instance-specific objectives if available
            if hasattr(metric, 'get_instance_objectives'):
                objectives.extend(metric.get_instance_objectives())
            else:
                objectives.extend(metric.get_objectives())
        return objectives

    def get_all_metrics(self):
        """Get all available metrics for post-hoc analysis"""
        metrics = []
        for metric in self.metrics:
            # Use instance-specific metrics if available
            if hasattr(metric, 'get_instance_metrics'):
                metrics.extend(metric.get_instance_metrics())
            else:
                metrics.extend(metric.get_metrics())
        return metrics

    def get_evaluation_metrics(self):
        """Get only evaluation metrics"""
        return [metric for metric in self.metrics if is_evaluation_metric(metric)]

    def get_training_metrics(self):
        """Get only training metrics"""
        return [metric for metric in self.metrics if is_training_metric(metric)]


def get_evaluator(config: dict) -> MetricEvaluator:
    """Factory function to create evaluator from config"""
    metrics = []
    if not isinstance(config, list):
        config = [config]
    for metric_config in config:
        metric_type = metric_config['name']
        if metric_type == 'wasserstein':
            metrics.append(WassersteinEvaluation())
        elif metric_type == 'percentile_classification':
            is_reversed = metric_config.get('reversed', False)
            if False and is_reversed:
                metrics.append(ReversedPercentileBasedIdOodClassifier(metric_config['threshold']))
            else:
                metrics.append(PercentileBasedClassifier(metric_config['threshold'], is_reversed))
        elif metric_type == 'tnr_at_tpr':
            metrics.append(TNRatTPX.from_config(metric_config))
        elif metric_type == 'runtime':
            metrics.append(BaseModelRuntimeEvaluation.from_config(metric_config))
        elif metric_type == 'uncertainty_estimating_runtime':
            metrics.append(UncertaintyEstimatingRuntimeEvaluation.from_config(metric_config))
        elif metric_type == 'mean_score':
            metrics.append(MeanScoreEvaluation())
        elif metric_type == 'max_score':
            metrics.append(MaxScoreEvaluation())
        elif metric_type == 'percentile_score':
            metrics.append(PercentileScoreEvaluation.from_config(metric_config))
        elif metric_type == 'base_model_throughput':
            metrics.append(BaseModelThroughputEvaluation.from_config(metric_config))
        elif metric_type == 'uncertainty_estimating_throughput':
            metrics.append(UncertaintyEstimatingThroughputEvaluation.from_config(metric_config))
        elif metric_type == 'auroc':
            metrics.append(AUROC())
        elif metric_type == 'max_memory_usage':
            metrics.append(MaxMemoryUsageEvaluation())
        elif metric_type == 'mape':
            metrics.append(MeanAbsolutePercentageError())
        elif metric_type == 'training_time':
            metrics.append(TrainingTimeMetric())
        # Add other metric types as needed
    
    return MetricEvaluator(metrics)


def get_uncertainty_evaluator(metric_config: str | dict | list) -> MetricEvaluator:
    """Factory function to create evaluator(s) from config
    
    Args:
        metric_config: Configuration for metrics in one of these formats:
            - string: naming a single metric
            - dict: with 'name' key and any required parameters for a single metric
            - list: of strings or dicts for multiple metrics
    
    Returns:
        MetricEvaluator containing the requested evaluation metric(s)
    """
    # Handle list input for multiple evaluators
    metrics = []
    
    if isinstance(metric_config, list):
        for config in metric_config:
            if isinstance(config, str):
                config = {'name': config}
            metrics.append(_create_single_evaluator(config))
    else:
        # Handle single metric (string or dict)
        if isinstance(metric_config, str):
            metric_config = {'name': metric_config}
        metrics.append(_create_single_evaluator(metric_config))
    
    return MetricEvaluator(metrics)

def _create_single_evaluator(metric_config: dict) -> EvaluationMetric:
    """Helper function to create a single evaluator from config"""
    distance_metrics = {
        WassersteinEvaluation.name: WassersteinEvaluation,
        EuclideanEvaluation.name: EuclideanEvaluation,
        JensenShannonEvaluation.name: JensenShannonEvaluation
    }

    name = metric_config['name']
    
    # Handle distance-based metrics
    if name in distance_metrics:
        return distance_metrics[name]()
    
    # Handle classification-based metrics
    if name == 'percentile_classification':
        threshold = metric_config['threshold']
        is_reversed = metric_config.get('reversed', False)
        return (ReversedPercentileBasedIdOodClassifier if is_reversed 
                else PercentileBasedIdOodClassifier)(threshold)
    elif name == 'tnr_at_tpr':
        target_tpr = metric_config['target_tpr']
        reversed = metric_config.get('reversed', False)
        return TNRatTPX(target_tpr, reversed)
    elif name == 'runtime':
        kwargs = {}
        if 'trials' in metric_config:
            kwargs['num_trials'] = metric_config['trials']
        if 'warmup' in metric_config:
            kwargs['num_warmup'] = metric_config['warmup']
        return BaseModelRuntimeEvaluation(**kwargs)
    elif name == 'uncertainty_estimating_runtime':
        return UncertaintyEstimatingRuntimeEvaluation()
    elif name == 'uncertainty_estimating_throughput':
        return UncertaintyEstimatingThroughputEvaluation.from_config(metric_config)
    elif name == 'mean_score':
        return MeanScoreEvaluation()
    elif name == 'max_score':
        return MaxScoreEvaluation()
    elif name == 'percentile_score':
        return PercentileScoreEvaluation.from_config(metric_config)
    elif name == 'auroc':
        return AUROC()
    elif name == 'mape':
        return MeanAbsolutePercentageError()
    elif name == 'training_time':
        return TrainingTimeMetric()
    else:
        raise ValueError(f"Invalid metric type: {name}")

class MeanAbsolutePercentageError(EvaluationMetric):
    name = "mape"

    def _calculate_mape(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Helper function to calculate Mean Absolute Percentage Error."""
        y_true_np = y_true.detach().cpu().numpy().flatten()
        y_pred_np = y_pred.detach().cpu().numpy().flatten()

        if y_true_np.shape[0] == 0:  # No samples
            return np.nan
            
        # Filter out entries where y_true is zero to avoid division by zero error
        # and to correctly represent MAPE (error relative to non-zero actuals)
        mask = y_true_np != 0
        
        y_true_filtered = y_true_np[mask]
        y_pred_filtered = y_pred_np[mask]
        
        if len(y_true_filtered) == 0: 
            # This case occurs if all true values were zero or the input was empty after filtering
            return np.nan 

        mape_val = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        return float(mape_val)

    def evaluate(self, model: nn.Module, id_data: tuple, ood_data: tuple) -> dict:
        model.eval()
        with torch.no_grad():
            id_inputs, id_true = id_data
            ood_inputs, ood_true = ood_data

            id_pred = model(id_inputs)
            ood_pred = model(ood_inputs)

            combined_inputs = torch.cat([id_inputs, ood_inputs], dim=0)
            combined_true = torch.cat([id_true, ood_true], dim=0)
            combined_pred = model(combined_inputs)
            
            mape_id = self._calculate_mape(id_true, id_pred)
            mape_ood = self._calculate_mape(ood_true, ood_pred)
            mape_combined = self._calculate_mape(combined_true, combined_pred)

        return {
            f'{self.name}_id': mape_id,
            f'{self.name}_ood': mape_ood,
            f'{self.name}_combined': mape_combined,
        }

    @classmethod
    def get_objectives(cls):
        return [
            {"name": f"{cls.name}_id", "type": "minimize"},
            {"name": f"{cls.name}_ood", "type": "minimize"},
            {"name": f"{cls.name}_combined", "type": "minimize"},
        ]

    @classmethod
    def get_metrics(cls):
        return [f'{cls.name}_id', f'{cls.name}_ood', f'{cls.name}_combined']

    def get_name(self):
        return self.name


def is_training_metric(metric: EvaluationMetric) -> bool:
    return isinstance(metric, TrainingMetric)

def is_evaluation_metric(metric: EvaluationMetric) -> bool:
    return isinstance(metric, EvaluationMetric)