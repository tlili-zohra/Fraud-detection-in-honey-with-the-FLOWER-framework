import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import flwr as fl
#import utils
from flwr_datasets import FederatedDataset
##utils.py
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from flwr.common import NDArrays, Metrics, Scalar


def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


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

####client.py

if __name__ == "__main__":
    N_CLIENTS = 10 #20 #3#10

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id
    #load dataset
    fds = FederatedDataset(dataset= "ZohraT/adulteration_dataset_26_08_2021.csv", partitioners={"train": 10})#3#10
    # Replace `partition_id` with the actual partition ID you want to load
    #partition_id = 0
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    L=[1 if i==0 else 0 for i in dataset['Concentration_Class']]
    S=pd.Series(L,name='Pur')
    NewData=pd.concat([dataset,S],axis=1)
    NewData=NewData.values
    # Assuming you have features and labels in your dataset
    X = NewData[:, 4:-2].astype(np.float32)
    y = NewData[:, -1].astype(np.float32)
    #xtrain, ytrain, xtest, ytest = train_test_split(X, y, test_size=0.2)
    # Assuming you want to split the data into train and test sets
    # 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
#initialiser le modele
    #max_iter= Ce paramètre définit le nombre maximum d'itérations pour que l'algorithme d'optimisation converge.
    model = LogisticRegression(
        penalty="l2",
        max_iter=100,  #Ancien max_iter=1 ,100,1000 #local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=X_train.shape[1], n_classes=3)
    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):
        #def __init__(self, model, X_train, y_train, X_test, y_test): #j'ajouter l'appel de notre initialisation
        #    self.model = model
        #    self.X_train, self.y_train = X_train, y_train
        #    self.X_test, self.y_test = X_test, y_test
        def get_parameters(self, config):  # type: ignore
            # Tracer les paramètres avant de les envoyer
            #coef, intercept = get_model_parameters(model)
            #print("Coefficients dans get_parametre:\n", coef)
            #print("Intercept dans get_parametre :\n", intercept)
            return get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                accuracy = model.score(X_train, y_train)
            # Tracer les paramètres avant de les envoyer
            #coef, intercept = get_model_parameters(model)
            #print("Coefficients dans fit :\n", coef)
            #print("Intercept dans fit :\n", intercept)
            return (get_model_parameters(model),len(X_train),
                    {"train_accuracy": accuracy},)
        def evaluate(self, parameters, config):  # type: ignore
            set_model_params(model, parameters)
            # Assuming y_test is your test target variable
            unique_labels = np.unique(y_test)
            loss = log_loss(y_test, model.predict_proba(X_test), labels=unique_labels)
            accuracy = model.score(X_test, y_test)
            # Tracer les paramètres apres de les envoyer ou les recevoir
            #coef, intercept = get_model_parameters(model)
            #print("Coefficients dans evaluate:\n", coef)
            #print("Intercept dans evaluate :\n", intercept)
            #print("----------------- End of round i ----------------- :\n")
            return loss, len(X_test), {"test_accuracy": accuracy}
    # Get the parameters
    #parameters = get_model_parameters(model)

    # Print the parameters
    #coef_, intercept_ = parameters
    #print("Coefficients client generale:\n", coef_)
    #print("Intercept client generale :\n", intercept_)
    #print("X_train:\n",X_train.shape[1])
    # Start Flower client
    fl.client.start_client(
        server_address="localhost:8080", client=FlowerClient().to_client()
        )
