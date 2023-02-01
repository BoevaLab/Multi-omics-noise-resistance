import argparse
import json
import os

import numpy as np
import pandas as pd
from noise_resistance.utils.factories import MODALITY_FACTORY
from noise_resistance.utils.misc_utils import (
    filter_modalities,
    get_noise_modalities,
    seed_torch,
)

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
    "--target",
    type=int,
    help="Whether one of the noise modalities should contain the observed event times.",
)

parser.add_argument(
    "--n_noise_dimensionality",
    type=int,
    help="Of what dimensionality the Gaussian noise to be added should be.",
)


def main(
    data_dir,
    config_path,
    modalities,
    n_noise_modalities,
    target,
    n_noise_dimensionality,
):
    n_noise_modalities = int(n_noise_modalities)
    n_noise_dimensionality = int(n_noise_dimensionality)
    with open(os.path.join(config_path, "config.json"), "r") as f:
        config = json.load(f)
    seed_torch(config.get("random_seed"))
    os.makedirs(os.path.join(data_dir, "processed/TCGA/noise"), exist_ok=True)
    for cancer in config["datasets"]:
        data_path = f"processed/TCGA/{cancer}_data_preprocessed.csv"
        data = pd.read_csv(
            os.path.join(data_dir, data_path),
            low_memory=False,
        ).drop(columns=["patient_id"])
        time, event = data["OS_days"].astype(int), data["OS"].astype(int)
        data = filter_modalities(
            data=data,
            selected_modalities_ix=modalities,
            all_modalities=config["modality_order"],
        )
        assert n_noise_modalities > 0

        noise_modalities = get_noise_modalities(
            n_noise_modalities=n_noise_modalities,
            time=time,
            noise_modality_dimensionality=n_noise_dimensionality,
        )
        data = pd.concat([data] + noise_modalities, axis=1)
        if target:
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        np.expand_dims(time.to_numpy(), 1),
                        columns=["clinical_target"],
                    ),
                ],
                axis=1,
            )
        data["OS_days"] = time
        data["OS"] = event
        os.makedirs(
            os.path.join(
                data_dir,
                "processed",
                "TCGA",
                cancer,
                "noise",
                f"{n_noise_modalities}",
                f"{'with' if target else 'without'}_target",
                f"{n_noise_dimensionality}_noise_dimensionality",
            ),
            exist_ok=True,
        )
        data.to_csv(
            os.path.join(
                data_dir,
                "processed",
                "TCGA",
                cancer,
                "noise",
                f"{n_noise_modalities}",
                f"{'with' if target else 'without'}_target",
                f"{n_noise_dimensionality}_noise_dimensionality",
                f"{MODALITY_FACTORY[modalities]}_preprocessed.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.data_dir,
        args.config_path,
        args.modalities,
        args.n_noise_modalities,
        args.target,
        args.n_noise_dimensionality,
    )
