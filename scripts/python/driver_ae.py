import argparse
import json
import os

import numpy as np
import pandas as pd
from noise_resistance.model.model import CoxPHNeural
from noise_resistance.utils.factories import (
    CRITERION_FACTORY,
    HYPERPARAM_FACTORY,
    MODALITY_FACTORY,
    SKORCH_NET_FACTORY,
)
from noise_resistance.utils.misc_utils import (
    StratifiedSurvivalKFold,
    drop_constants,
    filter_modalities,
    get_blocks,
    get_noise_modalities,
    negative_partial_log_likelihood_loss,
    seed_torch,
    transform_survival_target,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, help="Path to the folder containing data."
)
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the parameters needed for training in JSON format.",
)
parser.add_argument(
    "--results_path", type=str, help="Path where results should be saved"
)
parser.add_argument("--fusion", type=str, help="Name of model being trained.")

parser.add_argument(
    "--modalities",
    type=str,
    help="Which modalities are to be used for training - eg 0,1 indicates the first two modalities only are to be used. See config.json for details on modality ordering.",
)

parser.add_argument(
    "--n_noise_modalities",
    type=int,
    help="How many Gaussian noise modalities are to be added.",
)

parser.add_argument(
    "--noised_target",
    type=str,
    help="Whether one of the noise modalities should contain the observed event times plus Gaussian noise.",
)


def main(
    data_dir,
    config_path,
    results_path,
    fusion,
    modalities,
    n_noise_modalities,
    noised_target,
):
    save_here = os.path.join(results_path)
    os.makedirs(save_here, exist_ok=True)

    with open(os.path.join(config_path, "config.json"), "r") as f:
        config = json.load(f)
    noised_target = noised_target == "true"
    seed_torch(config.get("random_seed"))
    for cancer in config["datasets"]:
        data_path = (
            f"preprocessed/TCGA/{cancer}_data_complete_modalities_preprocessed.csv"
        )
        data = pd.read_csv(
            os.path.join(data_dir, data_path),
            low_memory=False,
        ).drop(columns=["patient_id"])
        time, event = data["OS_days"].astype(int), data["OS"].astype(int)
        data = data.drop(columns=["OS_days", "OS"])
        data = drop_constants(
            filter_modalities(
                data=data,
                selected_modalities_ix=modalities,
                all_modalities=config["modality_order"],
            )
        )
        train_splits = pd.read_csv(
            os.path.join(data_dir, f"splits/TCGA/{cancer}_train_splits.csv")
        )
        test_splits = pd.read_csv(
            os.path.join(data_dir, f"splits/TCGA/{cancer}_test_splits.csv")
        )
        if n_noise_modalities > 0:
            noise_modalities = get_noise_modalities(
                n_noise_modalities=n_noise_modalities,
                noised_target=noised_target,
                time=time,
                noise_modality_dimensionality=config[
                    "noise_modality_dimensionality"
                ],
            )
            data = pd.concat([data] + noise_modalities, axis=1)
        os.makedirs(
            os.path.join(
                save_here,
                "TCGA",
                cancer,
                f"{fusion}",
                f"{MODALITY_FACTORY[modalities]}",
                f"{n_noise_modalities}_noise_modalities_with{'' if noised_target else 'out'}_noised_target",
            ),
            exist_ok=True,
        )
        for outer_split in range(
            config["n_outer_splits"] * config["n_outer_repetitions"]
        ):
            print(outer_split)
            train_ix = (
                train_splits.iloc[outer_split, :].dropna().values.astype(int)
            )
            test_ix = (
                test_splits.iloc[outer_split, :].dropna().values.astype(int)
            )
            X_test = data.iloc[test_ix, :]
            X_train = data.iloc[train_ix, :]

            ct = ColumnTransformer(
                [
                    (
                        "numerical",
                        make_pipeline(StandardScaler()),
                        np.where(X_train.dtypes != "object")[0],
                    ),
                    (
                        "categorical",
                        make_pipeline(
                            OneHotEncoder(
                                sparse=False, handle_unknown="ignore"
                            ),
                            StandardScaler(),
                        ),
                        np.where(X_train.dtypes == "object")[0],
                    ),
                ]
            )
            y_train = transform_survival_target(
                time[train_ix].values, event[train_ix].values
            )
            X_train = ct.fit_transform(X_train)

            X_train = pd.DataFrame(
                X_train,
                columns=data.columns[
                    np.where(data.dtypes != "object")[0]
                ].tolist()
                + [
                    f"clinical_{i}"
                    for i in ct.transformers_[1][1][0]
                    .get_feature_names_out()
                    .tolist()
                ],
            )
            X_test = pd.DataFrame(
                ct.transform(X_test), columns=X_train.columns
            )

            net = SKORCH_NET_FACTORY[fusion](
                module=CoxPHNeural,
                criterion=CRITERION_FACTORY[fusion],
                module__fusion_method=fusion,
                module__blocks=get_blocks(X_train.columns),
            )
            net.set_params(
                **{
                    **HYPERPARAM_FACTORY["common_fixed"],
                    **HYPERPARAM_FACTORY[fusion],
                }
            )
            grid = GridSearchCV(
                net,
                {
                    **{
                        **HYPERPARAM_FACTORY["common_tuned"],
                        **HYPERPARAM_FACTORY["ae_tuned"],
                    }
                },
                scoring=make_scorer(
                    negative_partial_log_likelihood_loss,
                    greater_is_better=False,
                ),
                n_jobs=config["n_jobs"],
                refit=True,
                cv=StratifiedSurvivalKFold(n_splits=config["n_inner_splits"]),
            )

            grid.fit(
                X_train.to_numpy().astype(np.float32),
                y_train.astype(str),
            )
            survival_functions = (
                grid.best_estimator_.predict_survival_function(
                    X_test.to_numpy().astype(np.float32)
                )
            )
            survival_probabilities = np.stack(
                [
                    i(
                        np.unique(
                            pd.Series(y_train)
                            .str.rsplit("|")
                            .apply(lambda x: int(x[0]))
                            .values
                        )
                    )
                    .detach()
                    .numpy()
                    for i in survival_functions
                ]
            )

            sf_df = pd.DataFrame(
                survival_probabilities,
                columns=np.unique(
                    pd.Series(y_train)
                    .str.rsplit("|")
                    .apply(lambda x: int(x[0]))
                    .values
                ),
            )

            sf_df.to_csv(
                os.path.join(
                    save_here,
                    "TCGA",
                    cancer,
                    f"{fusion}",
                    f"{MODALITY_FACTORY[modalities]}",
                    f"{n_noise_modalities}_noise_modalities_with{'' if noised_target else 'out'}_noised_target",
                    f"split_{outer_split}.csv",
                ),
                index=False,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.data_dir,
        args.config_path,
        args.results_path,
        args.fusion,
        args.modalities,
        args.n_noise_modalities,
        args.noised_target,
    )
