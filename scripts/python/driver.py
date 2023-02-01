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
    seed_torch,
    transform_survival_target,
    negative_partial_log_likelihood_loss,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sksurv.nonparametric import kaplan_meier_estimator

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
    "--data",
    type=str,
    help="Which data type is being used. Must be one of 'regular', 'pca' or 'noise'..",
)


parser.add_argument(
    "--n_noise_modalities",
    type=int,
    help="How many Gaussian noise modalities are to be added.",
)


parser.add_argument(
    "--target",
    type=int,
    help="Whether one of the noise modalities should contain the observed event times plus Gaussian noise.",
)

parser.add_argument(
    "--n_noise_dimensionality",
    type=int,
    help="Of what dimensionality the Gaussian noise to be added should be.",
)

parser.add_argument(
    "--n_pca_dimensionality",
    type=int,
    help="Of what dimensionality the PCA reduction should be.",
)

parser.add_argument(
    "--pca_separate",
    type=str,
    help="Whether PCA should be run per modality or jointly.",
)

parser.add_argument(
    "--modality_dropout",
    type=int,
    help="Whether modality dropout should be applied (and tuned).",
)


def main(
    data_dir,
    config_path,
    results_path,
    fusion,
    modalities,
    input="regular",
    n_noise_modalities=0,
    target=0,
    n_noise_dimensionality=0,
    n_pca_dimensionality=0,
    pca_separate="separate",
    modality_dropout=False,
):
    save_here = os.path.join(results_path)
    os.makedirs(save_here, exist_ok=True)
    with open(os.path.join(config_path, "config.json"), "r") as f:
        config = json.load(f)
    seed_torch(config.get("random_seed"))
    for cancer in config["datasets"]:
        data_path = f"processed/TCGA/{cancer}_data_preprocessed.csv"
        data = (
            pd.read_csv(
                os.path.join(data_dir, data_path),
                low_memory=False,
            )
            .drop(columns=["patient_id"])
            .fillna("MISSING")
        )
        if input == "noise":
            data_path = f"processed/TCGA/{cancer}/noise/{n_noise_modalities}/{'with' if target else 'without'}_target/{n_noise_dimensionality}_noise_dimensionality/clinical_gex_preprocessed.csv"
        data = pd.read_csv(
            os.path.join(data_dir, data_path),
            low_memory=False,
        ).fillna("MISSING")
        if "patient_id" in data.columns:
            data = data.drop(columns=["patient_id"])
        time, event = data["OS_days"].astype(int), data["OS"].astype(int)
        data = data.drop(columns=["OS_days", "OS"])
        if input != "noise":
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
        for outer_split in range(
            config["n_outer_splits"] * config["n_outer_repetitions"]
        ):
            train_ix = (
                train_splits.iloc[outer_split, :].dropna().values.astype(int)
            )
            test_ix = (
                test_splits.iloc[outer_split, :].dropna().values.astype(int)
            )
            X_test = data.iloc[test_ix, :]
            X_train = data.iloc[train_ix, :]
            if "0" in modalities:
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
            else:
                ct = ColumnTransformer(
                    [
                        (
                            "numerical",
                            make_pipeline(StandardScaler()),
                            np.where(X_train.dtypes != "object")[0],
                        )
                    ]
                )
            y_train = transform_survival_target(
                time[train_ix].values, event[train_ix].values
            )
            X_train = ct.fit_transform(X_train)
            if "0" in modalities:
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
            else:
                X_train = pd.DataFrame(
                    X_train,
                    columns=data.columns[
                        np.where(data.dtypes != "object")[0]
                    ].tolist(),
                )
            X_test = pd.DataFrame(
                ct.transform(X_test), columns=X_train.columns
            )
            if input == "pca":
                pca = PCA(
                    # Set dimensionality as given dimensionality.
                    # If given dimensionality is too large (will only
                    # happen for `n_pca_dimensionality=128`), we use
                    # the smallest size of any training set.
                    n_components=min(n_pca_dimensionality, X_train.shape[0])
                )
                modality_list_train = []
                modality_list_test = []
                for modality in np.unique(
                    [col.rsplit("_")[0] for col in X_train.columns]
                ):
                    # Don't do PCA for clinical.
                    modality_columns = [
                        ix
                        for ix in range(X_train.shape[1])
                        if X_train.columns[ix].rsplit("_")[0] == modality
                    ]
                    if modality == "clinical":
                        modality_list_train.append(
                            X_train.iloc[:, modality_columns]
                        )
                        modality_list_test.append(
                            X_test.iloc[:, modality_columns]
                        )
                        continue

                    tmp_train = pd.DataFrame(
                        pca.fit_transform(X_train.iloc[:, modality_columns])
                    )
                    tmp_test = pd.DataFrame(
                        pca.transform(X_test.iloc[:, modality_columns])
                    )
                    tmp_train.columns = [
                        f"{modality}_{i}" for i in range(tmp_train.shape[1])
                    ]
                    tmp_test.columns = tmp_train.columns
                    modality_list_train.append(tmp_train)
                    modality_list_test.append(tmp_test)

                X_train = pd.concat(modality_list_train, axis=1)
                X_test = pd.concat(modality_list_test, axis=1)

            net = SKORCH_NET_FACTORY[fusion](
                module=CoxPHNeural,
                criterion=CRITERION_FACTORY[fusion],
                module__fusion_method=fusion,
                module__blocks=get_blocks(X_train.columns),
            )
            net.set_params(
                **{
                    **HYPERPARAM_FACTORY["common_fixed"],
                    **HYPERPARAM_FACTORY[f"{fusion}_fixed"],
                }
            )
            if modality_dropout:
                grid = GridSearchCV(
                    net,
                    {
                        **HYPERPARAM_FACTORY["common_tuned"],
                        **HYPERPARAM_FACTORY["multimodal_dropout"],
                        **HYPERPARAM_FACTORY[f"{fusion}_tuned"],
                    },
                    n_jobs=config["n_jobs_neural"],
                    cv=StratifiedSurvivalKFold(
                        n_splits=config["n_inner_splits"], shuffle=False
                    ),
                    scoring=make_scorer(
                        negative_partial_log_likelihood_loss,
                        greater_is_better=False,
                    ),
                    error_score=np.NINF,
                )
            else:
                grid = GridSearchCV(
                    net,
                    {
                        **HYPERPARAM_FACTORY["common_tuned"],
                        **HYPERPARAM_FACTORY[f"{fusion}_tuned"],
                    },
                    n_jobs=config["n_jobs_neural"],
                    cv=StratifiedSurvivalKFold(
                        n_splits=config["n_inner_splits"], shuffle=False
                    ),
                    scoring=make_scorer(
                        negative_partial_log_likelihood_loss,
                        greater_is_better=False,
                    ),
                    error_score=-np.NINF,
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
            if np.any(np.isnan(sf_df)):
                time_km, survival_km = kaplan_meier_estimator(
                    event[train_ix].values.astype(bool), time[train_ix].values
                )
                sf_df = pd.DataFrame(
                    [survival_km for i in range(X_test.shape[0])]
                )
                sf_df.columns = time_km

            if modalities == "0,1" and target and input == "regular":
                os.makedirs(
                    os.path.join(
                        save_here,
                        "survival_functions",
                        "TCGA",
                        cancer,
                        f"{fusion}",
                        f"{MODALITY_FACTORY[modalities]}",
                        "0_noise_modalities_with_target",
                    ),
                    exist_ok=True,
                )
                sf_df.to_csv(
                    os.path.join(
                        save_here,
                        "survival_functions",
                        "TCGA",
                        cancer,
                        f"{fusion}",
                        f"{MODALITY_FACTORY[modalities]}",
                        "0_noise_modalities_with_target",
                        f"split_{outer_split + 1}.csv",
                    ),
                    index=False,
                )
            elif input == "noise":
                os.makedirs(
                    os.path.join(
                        save_here,
                        "survival_functions",
                        "TCGA",
                        cancer,
                        f"{fusion}",
                        "clinical_gex",
                        "noise",
                        f"{n_noise_modalities}_noise_modalities_with{'' if target else 'out'}_target",
                        f"{n_noise_dimensionality}_noise_dimensions",
                    ),
                    exist_ok=True,
                )
                sf_df.to_csv(
                    os.path.join(
                        save_here,
                        "survival_functions",
                        "TCGA",
                        cancer,
                        f"{fusion}",
                        "clinical_gex",
                        "noise",
                        f"{n_noise_modalities}_noise_modalities_with{'' if target else 'out'}_target",
                        f"{n_noise_dimensionality}_noise_dimensions",
                        f"split_{outer_split + 1}.csv",
                    ),
                    index=False,
                )
            elif input == "pca":
                os.makedirs(
                    os.path.join(
                        save_here,
                        "survival_functions",
                        "TCGA",
                        cancer,
                        f"{fusion}",
                        f"{MODALITY_FACTORY[modalities]}",
                        "PCA",
                        pca_separate,
                        str(n_pca_dimensionality),
                    ),
                    exist_ok=True,
                )
                sf_df.to_csv(
                    os.path.join(
                        save_here,
                        "survival_functions",
                        "TCGA",
                        cancer,
                        f"{fusion}",
                        f"{MODALITY_FACTORY[modalities]}",
                        "PCA",
                        pca_separate,
                        str(n_pca_dimensionality),
                        f"split_{outer_split + 1}.csv",
                    ),
                    index=False,
                )
            else:
                if modality_dropout:
                    os.makedirs(
                        os.path.join(
                            save_here,
                            "survival_functions",
                            "TCGA",
                            cancer,
                            f"{fusion}",
                            f"{MODALITY_FACTORY[modalities]}",
                            "multimodal_dropout",
                        ),
                        exist_ok=True,
                    )

                    sf_df.to_csv(
                        os.path.join(
                            save_here,
                            "survival_functions",
                            "TCGA",
                            cancer,
                            f"{fusion}",
                            f"{MODALITY_FACTORY[modalities]}",
                            "multimodal_dropout",
                            f"split_{outer_split + 1}.csv",
                        ),
                        index=False,
                    )
                else:
                    os.makedirs(
                        os.path.join(
                            save_here,
                            "survival_functions",
                            "TCGA",
                            cancer,
                            f"{fusion}",
                            f"{MODALITY_FACTORY[modalities]}",
                        ),
                        exist_ok=True,
                    )
                    sf_df.to_csv(
                        os.path.join(
                            save_here,
                            "survival_functions",
                            "TCGA",
                            cancer,
                            f"{fusion}",
                            f"{MODALITY_FACTORY[modalities]}",
                            f"split_{outer_split + 1}.csv",
                        ),
                        index=False,
                    )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.n_noise_modalities is None:
        args.n_noise_modalities = 0
    if args.target is None:
        args.target = 0
    if args.n_noise_dimensionality is None:
        args.n_noise_dimensionality = 0
    if args.n_pca_dimensionality is None:
        args.n_pca_dimensionality = 0
    if args.pca_separate is None:
        args.pca_separate = "separate"
    if args.modality_dropout is None:
        args.modality_dropout = 0
    main(
        args.data_dir,
        args.config_path,
        args.results_path,
        args.fusion,
        args.modalities,
        args.data,
        int(args.n_noise_modalities),
        args.target,
        int(args.n_noise_dimensionality),
        int(args.n_pca_dimensionality),
        args.pca_separate,
        int(args.modality_dropout),
    )
