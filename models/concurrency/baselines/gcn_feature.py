import copy
import torch
import numpy as np
from typing import Optional, Tuple, List, Union


def featurize_queries_complex_online(
    existing_query_features: List[np.ndarray],
    existing_query_concur_features: List[Optional[torch.Tensor]],
    existing_pre_info_length: List[int],
    queued_query_features: List[np.ndarray],
    existing_start_time: List[float],
    current_time: float,
    next_finish_idx_list: Optional[Union[int, List[int]]] = None,
    next_finish_time_list: Optional[Union[float, List[float]]] = None,
    get_next_finish: bool = False,
    get_next_finish_running_performance: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    global_x = []
    global_pre_info_length = []
    if next_finish_idx_list is not None and type(next_finish_idx_list) != list:
        next_finish_idx_list = [next_finish_idx_list]
        next_finish_time_list = [next_finish_time_list]
    for query_feature in queued_query_features:
        l_feature = len(query_feature)
        x = []
        # concurrent feature for the current queued query
        if get_next_finish and next_finish_idx_list is not None:
            # the feature of the current query when the next running query finishes running
            next_finish_x = [[] for _ in range(len(next_finish_idx_list))]
        else:
            next_finish_x = None
        for i, exist_q in enumerate(existing_query_features):
            concur_query_feature = np.zeros(l_feature * 2 + 5)
            concur_query_feature[:l_feature] = query_feature
            concur_query_feature[(l_feature + 2) : (2 * l_feature + 2)] = exist_q
            concur_query_feature[l_feature] = 1
            concur_query_feature[2 * l_feature + 2] = (
                existing_start_time[i] - current_time
            )
            x.append(torch.FloatTensor(concur_query_feature))
            if next_finish_x is not None:
                for j in range(len(next_finish_idx_list)):
                    next_finish_idx = next_finish_idx_list[j]
                    next_finish_time = next_finish_time_list[j]
                    if i not in next_finish_idx_list[: (j + 1)]:
                        unfinished_concur_query_feature = copy.deepcopy(
                            concur_query_feature
                        )
                        unfinished_concur_query_feature[2 * l_feature + 2] = (
                            existing_start_time[i] - next_finish_time
                        )
                        next_finish_x[j].append(
                            torch.FloatTensor(unfinished_concur_query_feature)
                        )
        concur_query_feature = np.zeros(l_feature * 2 + 5)
        concur_query_feature[:l_feature] = query_feature
        x.append(torch.FloatTensor(concur_query_feature))
        global_pre_info_length.append(len(x))
        global_x.append(torch.stack(x))

        if next_finish_x is not None:
            for curr_next_finish_x in next_finish_x:
                if len(curr_next_finish_x) == 0:
                    # This can happen when the next finish query is the only query running in the system
                    concur_query_feature = np.zeros(l_feature * 2 + 5)
                    concur_query_feature[:l_feature] = query_feature
                    curr_next_finish_x.append(torch.FloatTensor(concur_query_feature))
                global_pre_info_length.append(len(curr_next_finish_x))
                global_x.append(torch.stack(curr_next_finish_x))

        # concurrent features for all existing (running queries) when this queued query is submitted
        for i in range(len(existing_query_concur_features)):
            global_pre_info_length.append(existing_pre_info_length[i])
            concur_query_feature = torch.zeros(l_feature * 2 + 5, dtype=torch.float)
            concur_query_feature[:l_feature] = torch.FloatTensor(
                existing_query_features[i]
            )
            concur_query_feature[(l_feature + 2) : (2 * l_feature + 2)] = (
                torch.FloatTensor(query_feature)
            )
            concur_query_feature[l_feature + 1] = 1.0
            concur_query_feature[2 * l_feature + 2] = (
                existing_start_time[i] - current_time
            )
            if existing_query_concur_features[i] is None:
                x = concur_query_feature.reshape(1, -1)
            else:
                x = torch.clone(existing_query_concur_features[i])
                x = torch.cat((x, concur_query_feature.reshape(1, -1)), dim=0)
            global_x.append(x)
            if (
                get_next_finish_running_performance
                and next_finish_idx_list is not None
                and len(next_finish_idx_list) != 0
            ):
                finished_query_features = []
                for j in range(len(next_finish_idx_list)):
                    next_finish_idx = next_finish_idx_list[j]
                    next_finish_time = next_finish_time_list[j]
                    finished_query_features.append(
                        existing_query_features[next_finish_idx]
                    )
                    if i not in next_finish_idx_list[: (j + 1)]:
                        concur_query_feature[2 * l_feature + 2] = (
                            existing_start_time[i] - next_finish_time
                        )
                        if existing_query_concur_features[i] is None:
                            x = concur_query_feature.reshape(1, -1)
                        else:
                            x = []
                            for k in range(len(existing_query_concur_features[i])):
                                k_query_feature = existing_query_concur_features[i][j][
                                    (l_feature + 2) : (2 * l_feature + 2)
                                ]
                                is_finished = False
                                for finished_query_feature in finished_query_features:
                                    if (
                                        not torch.sum(
                                            torch.abs(
                                                k_query_feature - finished_query_feature
                                            )
                                        )
                                        <= 1e-4
                                    ):
                                        # remove the finished query from its concurrent feature
                                        is_finished = True
                                        break
                                if not is_finished:
                                    x.append(existing_query_concur_features[i][j])
                            x.append(torch.FloatTensor(concur_query_feature))
                            x = torch.stack(x)
                    else:
                        # this query is already finished
                        x = torch.zeros((1, len(concur_query_feature)))
                    global_pre_info_length.append(max(len(x) - 1, 1))
                    global_x.append(x)
    global_pre_info_length = torch.LongTensor(global_pre_info_length)
    return global_x, global_pre_info_length
