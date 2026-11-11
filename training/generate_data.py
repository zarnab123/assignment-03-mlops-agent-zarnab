from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    dataset = load_breast_cancer()
    frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    frame["target"] = dataset.target
    frame = frame.iloc[: args.rows].copy()
    frame["dataset_version"] = args.version

    output_path = Path("data/breast_cancer.csv")
    output_path.parent.mkdir(exist_ok=True)
    frame.to_csv(output_path, index=False)

    print(
        f"Created {output_path} with {len(frame)} rows "
        f"for dataset version {args.version}"
    )


if __name__ == "__main__":
    main()
