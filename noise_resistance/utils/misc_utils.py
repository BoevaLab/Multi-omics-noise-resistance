import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from skorch.dataset import ValidSplit, get_len
from skorch.utils import to_numpy


# Adapted from https://github.com/pytorch/pytorch/issues/7068.
def seed_torch(seed=42):
    """Sets all seeds within torch and adjacent libraries.

    Args:
        seed: Random seed to be used by the seeding functions.

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


def transform_survival_target(time, event):
    return np.array([f"{time[i]}|{event[i]}" for i in range(len(time))])


def inverse_transform_survival_target(y):
    return (
        np.array([int(i.rsplit("|")[0]) for i in y]),
        np.array([int(i.rsplit("|")[1]) for i in y]),
    )


def create_risk_matrix(observed_survival_time):
    observed_survival_time = observed_survival_time.squeeze()
    return (
        (
            torch.outer(observed_survival_time, observed_survival_time)
            >= torch.square(observed_survival_time)
        )
        .long()
        .T
    )


def negative_partial_log_likelihood(
    predicted_log_hazard_ratio,
    observed_survival_time,
    observed_event_indicator,
):
    risk_matrix = create_risk_matrix(observed_survival_time)
    if torch.sum(observed_event_indicator) == 0:
        return torch.tensor(5.0, requires_grad=True)
    loss = -torch.sum(
        observed_event_indicator.float().squeeze()
        * (
            predicted_log_hazard_ratio.squeeze()
            - torch.log(
                torch.sum(
                    risk_matrix.float()
                    * torch.exp(predicted_log_hazard_ratio.squeeze()),
                    axis=1,
                )
            )
        )
    ) / torch.sum(observed_event_indicator)
    return loss


def get_noise_modalities(
    n_noise_modalities, noised_target, time, noise_modality_dimensionality
):
    noise_modalities = [
        np.random.normal(size=(time.shape[0], noise_modality_dimensionality))
        for ix in range(n_noise_modalities)
    ]
    if noised_target:
        noise_modalities[0] = np.expand_dims(
            time.to_numpy(), 1
        ) + np.random.standard_normal(np.expand_dims(time.to_numpy(), 1).shape)
    for ix in range(len(noise_modalities)):
        noise_modalities[ix] = pd.DataFrame(
            noise_modalities[ix],
            columns=[
                f"noise-{ix}_feature_{jx}"
                for jx in range(noise_modalities[ix].shape[1])
            ],
        )
    return noise_modalities


def calculate_log_hazard_input_size(fusion_method, blocks, modality_dimension):
    match fusion_method:
        case "early":
            return sum([len(block) for block in blocks])
        case "early_ae":
            return modality_dimension
        case "late_mean":
            raise ValueError
        case "late_moe":
            raise ValueError
        case "intermediate_mean":
            return modality_dimension
        case "intermediate_max":
            return modality_dimension
        case "intermediate_concat":
            return modality_dimension * len(blocks)
        case "intermediate_ae":
            return modality_dimension * len(blocks)
        case "intermediate_embrace":
            return modality_dimension
        case "intermediate_attention":
            return modality_dimension


def drop_constants(data):
    mask = (data != data.iloc[0]).any()
    return data.loc[:, mask]


def filter_modalities(data, selected_modalities_ix, all_modalities):
    modalities_to_keep_ix = np.array(
        [int(i) for i in selected_modalities_ix.rsplit(",")]
    )
    all_modalities = np.array(all_modalities)
    modalities_to_keep = all_modalities[modalities_to_keep_ix]
    modality_mask = [
        col for col in data.columns if col.rsplit("_")[0] in modalities_to_keep
    ]
    return data[modality_mask]


class StratifiedSkorchSurvivalSplit(ValidSplit):
    """Adapt `ValidSplit` to make it usable with our adapted
    survival target string format.

    For further documentation, please refer to the `ValidSplit`
    documentation, as the only changes made were to adapt the string
    target format.
    """

    def __call__(self, dataset, y=None, groups=None):
        if y is not None:
            # Handle string target by selecting out only the event
            # to stratify on.
            if y.dtype not in [np.dtype("float32"), np.dtype("int")]:
                y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                    np.float32
                )

        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y."
        )

        if (y is None) and self.stratified:
            raise bad_y_error

        cv = StratifiedKFold(n_splits=self.cv, random_state=42, shuffle=True)

        # pylint: disable=invalid-name
        len_dataset = get_len(dataset)
        if y is not None:
            len_y = get_len(y)
            if len_dataset != len_y:
                raise ValueError(
                    "Cannot perform a CV split if dataset and y "
                    "have different lengths."
                )

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid


def get_blocks(feature_names):
    column_types = (
        pd.Series(feature_names).str.rsplit("_").apply(lambda x: x[0]).values
    )
    return [
        np.where(
            modality
            == pd.Series(feature_names)
            .str.rsplit("_")
            .apply(lambda x: x[0])
            .values
        )[0].tolist()
        for modality in [
            q
            for q in ["clinical", "gex", "rppa", "mirna", "mut", "meth", "cnv"]
            + [f"noise-{ix}" for ix in range(10)]
            if q in np.unique(column_types)
        ]
    ]


class StratifiedSurvivalKFold(StratifiedKFold):
    """Adapt `StratifiedKFold` to make it usable with our adapted
    survival target string format.

    For further documentation, please refer to the `StratifiedKFold`
    documentation, as the only changes made were to adapt the string
    target format.
    """

    def _make_test_folds(self, X, y=None):
        if y is not None and isinstance(y, np.ndarray):
            # Handle string target by selecting out only the event
            # to stratify on.
            if y.dtype not in [np.dtype("float32"), np.dtype("int")]:
                y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                    np.float32
                )

        return super()._make_test_folds(X=X, y=y)

    def _iter_test_masks(self, X, y=None, groups=None):
        if y is not None and isinstance(y, np.ndarray):
            # Handle string target by selecting out only the event
            # to stratify on.

            if y.dtype not in [np.dtype("float32"), np.dtype("int")]:
                y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                    np.float32
                )
            else:
                event = y[:, 1]
                return super()._iter_test_masks(X, y=event)
        return super()._iter_test_masks(X, y=y)

    def split(self, X, y, groups=None):
        return super().split(X=X, y=y, groups=groups)


def negative_partial_log_likelihood_loss(
    y_true,
    y_pred,
):
    (
        observed_survival_time,
        observed_event_indicator,
    ) = inverse_transform_survival_target(y_true)
    return negative_partial_log_likelihood(
        y_pred,
        torch.tensor(observed_survival_time),
        torch.tensor(observed_event_indicator),
    )


def negative_partial_log_likelihood_loss_multi_output(
    y_true,
    y_pred,
):
    (
        observed_survival_time,
        observed_event_indicator,
    ) = inverse_transform_survival_target(y_true)
    return negative_partial_log_likelihood(
        y_pred[0],
        torch.tensor(observed_survival_time),
        torch.tensor(observed_event_indicator),
    )
