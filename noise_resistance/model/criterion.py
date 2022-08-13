import torch
from noise_resistance.utils.misc_utils import (
    negative_partial_log_likelihood,
)


class cox_ph_criterion(torch.nn.Module):
    def forward(self, predicted, target):
        if isinstance(predicted, tuple) and len(predicted > 1):
            predicted = predicted[0]
        return negative_partial_log_likelihood(
            predicted,
            target[:, 0],
            target[:, 1],
        )


class cox_ae_criterion(torch.nn.Module):
    def forward(self, predicted, target, alpha):
        cox_loss = alpha * negative_partial_log_likelihood(
            predicted[0],
            target[:, 0],
            target[:, 1],
        )
        reconstruction_loss = torch.nn.MSELoss()(predicted[1], predicted[2])
        return cox_loss + reconstruction_loss
