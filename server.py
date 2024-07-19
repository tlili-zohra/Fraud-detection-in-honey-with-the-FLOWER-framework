import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from typing import List, Tuple, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

from flwr.common import NDArrays, Metrics, Scalar


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    print(metrics)
    # num_samples_list can represent number of sample or batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    return weighted_metrics




# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    set_initial_params(model, n_classes=2, n_features=4) #n_classes=3,n_features=4
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=5,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )#2,3,4,5 min client
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=25),
    )
