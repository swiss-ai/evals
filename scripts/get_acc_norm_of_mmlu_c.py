import json
import numpy as np

RESULT_PATH = (
    '/iopsstor/scratch/cscs/mzueri/eval-logs/cooldown_hplt4_dclm_10pct/iter_0050000/harness/eval_3058165/'
    '__iopsstor__scratch__cscs__mzueri__.tmp__tmp.4LISozJl5Q/'
    'results_2026-08-11T19-50-13.332905.json'
)



def compute_subtask_average(all_results, n_samples, group_bench, metric_key):
    """
    Computes the weighted average of `metric_key` across all subtasks of
    `group_bench`, weighted by the effective sample count from `n_samples`.
    """
    prefix = f"{group_bench}_"
    subtask_values = []
    subtask_weights = []

    for key, data in all_results.items():
        if not key.startswith(prefix):
            continue
        val = None
        if f"{metric_key},none" in data:
            val = data[f"{metric_key},none"]
        elif metric_key in data:
            val = data[metric_key]
        if val is not None:
            weight = 1.0
            if n_samples and key in n_samples:
                weight = float(
                    n_samples[key].get(
                        "effective", n_samples[key].get("original", 1.0)
                    )
                )
            subtask_values.append(val)
            subtask_weights.append(weight)

    if not subtask_values:
        return None

    values = np.array(subtask_values)
    weights = np.array(subtask_weights)
    avg = float(np.average(values, weights=weights))

    is_weighted = n_samples is not None and any(w != 1.0 for w in subtask_weights)
    avg_type = "weighted" if is_weighted else "unweighted"

    return avg

subtask_average = compute_subtask_average(json.loads(open(RESULT_PATH).read())['results'], None, 'mmlu_continuation', 'acc_norm')
print(f"Subtask average for mmlu_continuation: {subtask_average}")