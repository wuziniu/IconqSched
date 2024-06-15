import pandas as pd
import numpy as np


def remove_long_idle_interval(trace_path: str, threshold: int = 400, inplace: bool = True) -> pd.DataFrame:
    """
    Remove long idle interval in the trace that has no query running
    Doing this can improve the efficiency of running experiments
    """
    trace = pd.read_csv(trace_path)
    trace = trace.sort_values(by=["g_offset_since_start_s"], ascending=True)

    new_start_time = []
    start_s = trace["g_offset_since_start_s"].values
    prev_start_time = 0
    offset = 0
    for i in range(len(trace)):
        if start_s[i] > prev_start_time + threshold:
            offset += (start_s[i] - prev_start_time - threshold)
        new_start_time.append(start_s[i] - offset)
        prev_start_time = start_s[i]
    trace["g_offset_since_start_s"] = np.asarray(new_start_time)
    if inplace:
        trace.to_csv(trace_path, index=False)
    return trace
