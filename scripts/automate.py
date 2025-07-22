"""Automatic evaluations
Usage (in a tmux session that never ends):
```
while true; do
    python scripts/automate.py
    sleep $(( 60*60 ))
done
```
Make sure to export your WANDB_API_KEY and LOGS_ROOT.
"""
from __future__ import annotations

import collections
import re
import os
import math
import json
import subprocess
import shutil
from pathlib import Path

from evals.tasks import Task, get_all_tasks, get_partition


def get_running(as_jobname: bool = False) -> dict[str, dict[int, list[str]] | list[str]]:
    proc = subprocess.run(["squeue", "--me", '--format="%j"', "--noheader"],
                          capture_output=True, text=True)
    assert proc.returncode == 0

    jobnames = proc.stdout.strip().split("\n")
    if jobnames == [""]:
        return [] if as_jobname else collections.defaultdict(lambda: collections.defaultdict(list))
    jobnames = [re.match('^"(.*)"$', jobname).group(1) for jobname in jobnames]

    running = [] if as_jobname else collections.defaultdict(lambda: collections.defaultdict(list))
    for jobname in jobnames:
        rmatch = re.match(r"^eval_(.*)_([a-zA-Z]+)_([0-9]+)$", jobname)
        if rmatch is not None:
            if as_jobname:
                running.append(jobname)
            else:
                name, group, it = rmatch.groups()
                running[name][int(it)] += group
    return running


def get_evaluated(model: str) -> dict[int, list[str]]:
    status = collections.defaultdict(list)
    for path in Path(CFG["logs_root"]).glob(f"{model}/iter_*/harness/eval_*/*/results*.json"):
        it = int(re.match("^iter_([0-9]+)$", path.parent.parent.parent.parent.name).group(1))
        with open(path) as f:
            info = json.load(f)
        for taskname in info["results"]:
            status[it].append(taskname)
    return status


def get_available(model_dirs: list[Path]) -> list[int]:
    available = []
    for model_dir in model_dirs:
        for path in filter(lambda path: path.suffix == "", Path(model_dir).iterdir()):
            available.append(int(re.match("^iter_([0-9]+)$", path.name).group(1)))
    return available


def submit(name: str, model: dict, it: int, tasks: list[Task]):
    # Get partition of tasks.
    total_size = sum(task.size for task in ALL_TASKS)
    n_shards = math.ceil(total_size/model["max_samples"])
    partition = get_partition(tasks=tasks, shards=n_shards)
    default_partition = get_partition(tasks=ALL_TASKS, shards=n_shards)

    # Schedule all tasks requested.
    path, = (model_dir for model_dir in model["model_dirs"]
             if Path(f"{model_dir}/iter_{it:07d}").exists())
    for part in partition:
        # Get special jobname depending on the tasks requested.
        matches = [(i, default_part) for i, default_part in enumerate(default_partition)
                   if part == default_part]
        if len(matches) == 0:
            jobname = "mixed"
        else:
            (shard_i, _), = matches
            jobname = f"shard{shard_i}of{n_shards}"
        jobname = f"eval_{name}_{jobname}_{it}"

        cmd = ["sbatch", f"--job-name={jobname}", "scripts/evaluate.sbatch", str(path),
               str(it), model["tokens_per_iter"], name] 
        env = {**os.environ,
               "LOGS_ROOT": CFG["logs_root"],
               "TOKENIZER": "alehc/swissai-tokenizer",
               "BOS": "true",
               "SIZE": str(model["size"]),
               "HF_TEMP_DIR": CFG["hf_temp_dir"],
               "TASKS": ",".join(task.name for task in part)}
        print("Launching", jobname)
        subprocess.run(cmd, env=env, stdout=subprocess.PIPE)


def submit_needed():
    running = get_running()
    for name, model in CFG["models"].items():
        total_size = sum(task.size for task in ALL_TASKS)
        n_shards = math.ceil(total_size/model["max_samples"])
        default_partition = get_partition(tasks=ALL_TASKS, shards=n_shards)

        # Get tasks alredy evaluated (reading them from the `results.json`).
        status = get_evaluated(name)
        default_partition = get_partition(tasks=ALL_TASKS, shards=n_shards)

        # Handle already evaluated: if a "mixed" group is running, assume it will
        # contain all missing tasks because we don't know which one does it contain in reality,
        # otherwise obtain the correct shard.
        for it, groups in running[name].items():
            for group in groups:
                if groups == "mixed":
                    actual_tasks = ALL_TASKS
                else:
                    shard_i, total_shards = re.match("^shard([0-9]+)of([0-9]+)$", group).groups()
                    assert total_shards == n_shards
                    actual_tasks = default_partition[int(shard_i)]

                if it in status:
                    status[it] += [task.name for task in actual_tasks]
                else:
                    status[it] = [task.name for task in actual_tasks]

        available = get_available(model["model_dirs"])
        for it in available:
            if (it - model["start_eval_from"]) % model["frequency"] == 0 and it >= model["start_eval_from"]:
                # Determine missing set.
                missing = []
                handled = status.get(it, [])
                for task in ALL_TASKS:
                    if len(task.alias) > 0 and any(actual_name not in handled for actual_name in task.alias):
                        missing.append(task)
                    elif len(task.alias) == 0 and task.name not in handled:
                        missing.append(task)
                if len(missing) > 0:
                    submit(name, model, it, missing)


def update_hf_checkpoints():
    jobnames = get_running(as_jobname=True)
    for path in Path(CFG["hf_temp_dir"]).iterdir():
        if path.name in jobnames:  # Don't touch hf checkpoints of unfinished runs.
            continue
        name, it = re.match("^eval_(.*)_.*_([0-9]+)$", path.name).groups()
        it = int(it)
        dest = Path(CFG["hf_storage_dir"])/f"{name}_it{it}"
        if dest.exists():  # Checkpoint is already stored, probably from a job with different tasks that finished earlier.
            print("Removing", path)
            shutil.rmtree(path)
        else:
            print("Moving", path, "to", dest)
            shutil.move(path, dest)


def cleanup_hf_checkpoints():
    # Get model=>[(it, path)] mapping.
    stored = collections.defaultdict(list)
    for path in Path(CFG["hf_storage_dir"]).iterdir():
        rmatch = re.match("^(.*)_it([0-9]+)$", path.name)
        if rmatch is not None and rmatch.group(1) in CFG["models"]:
            name, it = rmatch.groups()
            stored[name].append((int(it), path))

    # Remove old checkpoints.
    keep = CFG["num_hf_checkpoints_to_keep"]
    for saved in stored.values():
        remove = sorted(saved, key=lambda t: t[0])[:-keep]
        for _, path in remove:
            print("Removing", path)
            shutil.rmtree(path)


def sync_wandb():
    print("Syncing wandb...")
    env = {**os.environ,
           "WANDB_SILENT": "true",
           "WANDB_RESUME": "allow",
           "WANDB_ENTITY": CFG["wandb_entity"],
           "WANDB_PROJECT": CFG["wandb_project"]}
    cmd = ["python3", "scripts/update_wandb.py", str(CFG["logs_root"])]
    for name in CFG["models"]:
        subprocess.run(cmd + [f"--name={name}"], env=env)


def main():
    submit_needed()
    update_hf_checkpoints()
    cleanup_hf_checkpoints()
    sync_wandb()


if __name__ == "__main__":
    ALL_TASKS = get_all_tasks()
    with open("configs/automation.json") as f:
        CFG = json.load(f)
    main()
