import numpy as np
import torch
from noise_resistance.imports.sksurv_imports import BreslowEstimator
from noise_resistance.utils.misc_utils import inverse_transform_survival_target, seed_torch
from skorch.callbacks import Callback
from skorch.net import NeuralNet
from skorch.utils import to_tensor


class CoxPHNeuralNet(NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        time, event = inverse_transform_survival_target(y_true)
        time = to_tensor(time, device="cpu")
        event = to_tensor(event, device="cpu")
        y_true = torch.stack([time, event], axis=1)
        return self.criterion_(y_pred, y_true)

    def forward(self, X, training=False):
        y_infer = list(self.forward_iter(X, training=training))
        return y_infer[0]

    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()
        time, event = inverse_transform_survival_target(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(
            np.array(
                self.module_.forward(torch.tensor(X)).detach().numpy().ravel()
            ),
            time,
            event,
        )
        return self

    def fit_breslow(self, log_hazard_ratios, time, event):
        self.breslow = BreslowEstimator().fit(log_hazard_ratios, event, time)

    def predict_survival_function(self, X):
        log_hazard_ratios = self.forward(X)
        survival_function = self.breslow.get_survival_function(
            log_hazard_ratios
        )
        return survival_function

    def predict(self, X):
        log_hazard_ratios = self.forward(X)
        return log_hazard_ratios


class CoxPHAENeuralNet(CoxPHNeuralNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        time, event = inverse_transform_survival_target(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(
            self.module_.forward(torch.tensor(X))[0].detach().numpy().ravel(),
            time,
            event,
        )
        return self

    def get_loss(self, y_pred, y_true, X=None, training=False):
        time, event = inverse_transform_survival_target(y_true)
        time = to_tensor(time, device="cpu")
        event = to_tensor(event, device="cpu")
        y_true = torch.stack([time, event], axis=1)
        return self.criterion_(
            predicted=y_pred, target=y_true, alpha=self.module_.alpha
        )

    def predict_survival_function(self, X):
        log_hazard_ratios = self.forward(X)[0]
        survival_function = self.breslow.get_survival_function(
            log_hazard_ratios
        )
        return survival_function

    def predict(self, X):
        log_hazard_ratios = self.forward(X)[0]
        return log_hazard_ratios


class FixSeed(Callback):
    def __init__(self, seed):
        self.seed = seed

    def initialize(self):
        seed_torch(self.seed)
        return super().initialize()
