import argparse
from pathlib import Path

import yaml

from evals.tasks import get_all_tasks


def main(config: Path, out: Path):
    info = {
        "group": "swissai_eval",
        "tasks": sorted([task.name for task in get_all_tasks(all_tasks_json=config)]),
        "aggregate_metric_list": [
            {"metric": "acc", "aggregation": "mean", "weight_by_size": False},
            {"metric": "acc_norm", "aggregation": "mean", "weight_by_size": False},
            {"metric": "perplexity", "aggregation": "mean", "weight_by_size": False},
            {"metric": "f1", "aggregation": "mean", "weight_by_size": False},
            {"metric": "exact_match", "aggregation": "mean", "weight_by_size": False},
        ],
        "metadata": {"version": 1.1},
    }
    with open(out, "w+") as f:
        yaml.dump(info, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/all_tasks.json"))
    parser.add_argument("--out", type=Path, default=Path("default.yaml"))
    main(**vars(parser.parse_args()))
