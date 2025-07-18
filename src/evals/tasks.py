from __future__ import annotations

import dataclasses
import enum
import json
import os
from pathlib import Path

import requests
import iso639


REQUEST_CACHE = {}


class Dimension(enum.StrEnum):
    general_abilities = enum.auto()
    factual_agnostic = enum.auto()
    factual_regional = enum.auto()

    @classmethod
    def get(cls, name: str) -> Dimension:
        general = ["hellaswag", "piqa", "arc", "ai2_arc", "winogrande", "xwinograd", "xnli", "copa", "xcopa"]
        agnostic = ["mmlu", "global_mmlu"]
        regional = ["include", "switzerland_qa"]

        if any(name.startswith(group) for group in general):
            return Dimension.general_abilities
        if any(name.startswith(group) for group in agnostic):
            return Dimension.factual_agnostic
        if any(name.startswith(group) for group in regional):
            return Dimension.factual_regional
        raise ValueError(f"Could not infer dimension for task {task}")


class TaskKind(enum.StrEnum):
    pretrain = enum.auto()
    posttrain = enum.auto()


@dataclasses.dataclass
class Task:
    name: str
    kinds: tuple[TaskKind]
    size: int = None
    language: iso639.Lang = None
    dimension: Dimension = None

    def __hash__(self) -> int:
        return hash((self.name, self.kinds, self.size, self.language.pt1, self.dimension))

    def __post_init__(self):
        # Infer size, language and dimension if not specified.
        if self.size is None:
            self.size = _infer_size(self.name)
        if self.language is None:
            chunks = self.name.split("_")
            if len(chunks) == 1:  # no language code, assume English.
                self.language = iso639.Lang("en")
            elif len(chunks[-1]) == 2:  # iso639 already expected.
                self.language = iso639.Lang(chunks[-1])
            else:  # full name of the language given.
                self.language = iso639.Lang(chunks[-1].title())
        if self.dimension is None:
            self.dimension = Dimension.get(self.name)

def _infer_size(name: str) -> int:
    def query(source: str) -> dict:
        headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
        url = f"https://datasets-server.huggingface.co/info?dataset={source}"
        response = requests.get(url, headers=headers)
        return response.json()

    exact_sources = {
        "hellaswag": ("Rowan/hellaswag", "default"),
        "mmlu": ("cais/mmlu", None),
        "winogrande": ("allenai/winogrande", "winogrande_xl"),
        "ai2_arc": ("allenai/ai2_arc", None),
    }

    underscore_sources = {
        "arc": "alexandrainst/m_arc",
        "global_mmlu": "CohereLabs/Global-MMLU",
        "hellaswag": "alexandrainst/m_hellaswag",
        "include_base_44": "CohereLabs/include-base-44",
        "xcopa": "cambridgeltl/xcopa",
        "xnli": "facebook/xnli",
        "xwinograd": "Muennighoff/xwinograd",
    }

    if name in exact_sources:
        source, config = exact_sources[name]
    elif "_" in name:
        root = "_".join(name.split("_")[:-1])
        if root not in underscore_sources:
            raise ValueError(f"Could not infer size for task {name}")
        source = underscore_sources[root]
        config = name.split("_")[-1]
    else:
        raise ValueError(f"Could not infer size for task {name}")

    if name.startswith("include_base_44"):
        config = config.title()

    if source not in REQUEST_CACHE:
        REQUEST_CACHE[source] = query(source)

    if config is None:
        all_splits = [subset["splits"]
                      for subset in REQUEST_CACHE[source]["dataset_info"].values()]
    else:
        all_splits = [REQUEST_CACHE[source]["dataset_info"][config]["splits"]]
    return sum(split["num_examples"]
               for splits in all_splits for split in splits.values())




def get_all_tasks(all_tasks_json: Path = Path("configs/all_tasks.json")) -> list[Task]:
    with open(all_tasks_json) as f:
        raw_tasks = json.load(f)
    
    tasks = []
    for kind, names in raw_tasks["infer"].items():
        for name in names:
            tasks.append(Task(name, (kind,)))
    for row in raw_tasks["other"]:
        tasks.append(Task(row["name"], tuple(row["kinds"]), row["size"],
                          None if row["language"] is None else iso639.Lang(row["language"]), row["dimension"]))
    return tasks
