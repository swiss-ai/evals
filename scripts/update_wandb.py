import collections
import statistics
import re
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import wandb
import iso639

from evals.tasks import get_all_tasks, Task


def get_log(infos: List[dict], tasks_cfg: dict, all_tasks: list[Task]) -> dict[str, float]:
    def agg(log: dict[str, dict[str, float]], prefix: str, tasks_to_agg: list[str], warn: bool = True):
        missing = set(tasks_to_agg) - set(log)
        if len(missing) > 0:
            if warn:
                print("WARNING! Macro aggregation for", prefix, "not available. Missing:", sorted(missing))
            return

        for metric in filter(lambda metric: "stderr" not in metric, all_metrics):
            values = [log[taskname][metric] for taskname in tasks_to_agg if metric in log[taskname]]
            if len(values) > 0:
                log[f"{prefix}.macro"][metric] = statistics.mean(values)

    # Aggregate raw info.
    groups = {}
    results = {}
    all_metrics = set()
    for info in infos:
        groups.update(info["group_subtasks"])
        results.update(info["results"])

    # Prepare final logs.
    log = collections.defaultdict(dict)
    for dataname, details in results.items():
        for metricname, val in details.items():
            if metricname == "alias" or val in ["N/A", " "]:
                continue
            assert isinstance(val, float), f"{dataname}.{metricname} = val"
            metricname, _ = metricname.split(",")  # for some reason it is always acc,none so we remove the none.
            all_metrics.add(metricname)
            log[dataname][metricname] = val

    # Now that we have all the "leaf task groups" we can do four aggregations:
    # Let's start with the {language_group} agg.
    for lang_group_name, langs in tasks_cfg["language_groups"].items():
        tasks_to_agg = [task.name for task in all_tasks
                        if task.language.pt1 in langs]
        agg(log, f"language_group/{lang_group_name}", list(tasks_to_agg))

    # Aggregate start with the {dimension} agg.
    all_dims = sorted({task.dimension for task in all_tasks})
    for dim in all_dims:
        tasks_to_agg = [task.name for task in all_tasks
                        if task.dimension == dim]
        agg(log, f"dimension/{dim}", list(tasks_to_agg))


    # Agregate {dimension}.{language_group}.macro.
    for lang_group_name, langs in tasks_cfg["language_groups"].items():
        for dim in all_dims:
            tasks_to_agg = [task.name for task in all_tasks
                            if task.language.pt1 in langs and task.dimension == dim]
            agg(log, f"dimension_group/{dim}.{lang_group_name}", list(tasks_to_agg))

    # Finally, {dimension}.{language}
    all_langs = sorted({task.language.pt1 for task in all_tasks})
    for lang in all_langs:
        for dim in all_dims:
            tasks_to_agg = [task.name for task in all_tasks
                            if task.language.pt1 == lang and task.dimension == dim]
            agg(log, f"dimension_lang/{dim}.{lang}", list(tasks_to_agg))

    # Finally, prepare wandb format.
    wandb_log = {}
    for dataname, details in log.items():
        for metric, value in details.items():
            wandb_log[f"{dataname}/{metric}"] = value
    return wandb_log


def get_history(name: str) -> Dict[int, Dict[str, float]]:
    api = wandb.Api()
    try:
        run = api.run(f"{api.default_entity}/{os.environ['WANDB_PROJECT']}/{name}")
    except wandb.errors.errors.CommError:  # Run not found.
        return {}
    history = collections.defaultdict(dict)
    for row in run.scan_history():
        row = {key: value for key, value in row.items() if not key.startswith("_")}
        history[row["ConsumedTokens"]] = row
    return history


def repair(all_tasks: list[Task]) -> list[Task]:
    repaired = []
    for task in all_tasks:
        if task.name == "ai2_arc":
            repaired += [Task("arc_easy", (), 0, iso639.Lang("en"), task.dimension),
                         Task("arc_challenge", (), 0, iso639.Lang("en"), task.dimension)]
        else:
            repaired.append(task)
    return repaired


def main(logs_root: Path, name: Optional[str], it: Optional[int], cfg: Path):

    all_tasks = get_all_tasks(all_tasks_json=cfg/"all_tasks.json")
    all_tasks = repair(all_tasks)
    with open(cfg/"tasks.json") as f:
        tasks_cfg = json.load(f)
    all_languages = {task.language.pt1 for task in all_tasks}

    for lang_group in tasks_cfg["language_groups"].values():
        for lang in lang_group:
            assert lang in all_languages or lang == "rm", lang

    tasks_cfg["language_groups"]["global"] = list(all_languages)
    tasks_cfg["language_groups"]["multilingual"] = list(all_languages - {"en"})

    # Grab each possible log and update wandb run.
    # First, iterate model names.
    latest_logs = {}
    for p1 in filter(lambda p: name is None or name == p.name, logs_root.iterdir()):
        print("Updating path", p1)
        history = get_history(p1.name)  # Get already pushed information.
        with wandb.init(id=p1.name, name=p1.name) as run:
            run.define_metric("ConsumedTokens")
            run.define_metric("*", step_metric="ConsumedTokens")
            # Now iterate all iterations for this name.
            for p2 in p1.iterdir():
                current_it = int(re.match("^iter_([0-9]+)$", p2.name).group(1))
                # Skip if specified --it doesn't match currently iterated it.
                if it is not None and it != current_it:
                    continue
                print("Updating iteration", current_it)

                with open(p2/"consumed_tokens.txt") as f:
                    consumed_tokens = int("".join(f).strip())

                # Get all results.json harness logs.
                results = []
                for path in sorted(p2.glob("harness/eval_*/*/results*.json")):
                    with open(path) as f:
                        results.append(json.load(f))

                if len(results) > 0:
                    log = get_log(results, tasks_cfg, all_tasks)
                    log.update({"ConsumedTokens": consumed_tokens, "OptStep": current_it})
                    # Update log if needed.
                    if consumed_tokens in history:
                        if "eval_table" in history[consumed_tokens]:
                            del history[consumed_tokens]["eval_table"]
                        if log == history[consumed_tokens]:
                            print("Exact log already matches wandb! Ignoring entry to avoid pushing duplicates")
                        else:
                            print(sorted(set(history[consumed_tokens]) - set(log)))
                            print("Important! wandb log at current iteration already found, but differs. Updating")
                            run.log(log)
                    else:
                        run.log(log)

                    # Update all_logs so we can build the table after this big loop.
                    if p1.name not in latest_logs or latest_logs[p1.name]["ConsumedTokens"] < consumed_tokens:
                        latest_logs[p1.name] = log
                else:
                    print("No logs found!")
                print()

    # Build and push the table.
    # We need `it` to be None to ensure that the logs on `latest_logs` actually
    # belong to the latest known iteration.
    show_in_table = tasks_cfg["show_in_table"]
    if it is None:
        for name, log in filter(lambda t: set(show_in_table) <= set(t[1]),
                                latest_logs.items()):
            print("Updating table for model", name)
            sublog = {"Model": name}
            sublog.update({task: log[task] for task in ["ConsumedTokens"] + show_in_table})
            df = pd.DataFrame([sublog])
            with wandb.init(id=name, name=name) as run:
                run.log({"eval_table": wandb.Table(dataframe=df), "ConsumedTokens": log["ConsumedTokens"]})
    print("Goodbye")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logs_root", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--it", type=int)
    parser.add_argument("--cfg", type=Path, default=Path("configs"))
    args = parser.parse_args()
    main(**vars(args))
