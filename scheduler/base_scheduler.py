import pandas as pd
import numpy as np
import torch
from typing import Optional, Tuple, List, Union, MutableMapping
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN


def reverse_index_list(lst: List, pop_index: List[int]) -> List:
    return [
        lst[i] for i in range(len(lst)) if i not in pop_index
    ]


class BaseScheduler:
    def __init__(
        self,
        stage_model: SingleStage,
        predictor: ConcurrentRNN,
        max_concurrency_level: int = 10,
        min_concurrency_level: int = 2
    ):
        """
        The class for basic scheduler with a naive scheduling algorithm
        :param stage_model: the staged single query runtime prediction model
        :param predictor: the LSTM based runtime predictor for concurrent queries
        :param max_concurrency_level: the maximum number of concurrently running queries a system can take
        :param min_concurrency_level: when there are less than min_concurrency_level queries running,
                                      can consider the system to be underloaded
        TODO: instead of definding system under/overload through the number of concurrently running queries,
              we can define based on predicted runtime
        """
        self.stage_model = stage_model
        self.predictor = predictor
        self.max_concurrency_level = max_concurrency_level
        self.min_concurrency_level = min_concurrency_level

        self.existing_query_features: List[np.ndarray] = []
        self.existing_query_concur_features: List[Optional[torch.Tensor]] = []
        self.existing_pre_info_length: List[int] = []
        self.existing_start_time: List[float] = []
        self.existing_finish_time: List[float] = []
        self.existing_runtime_prediction_dict: MutableMapping[Union[str, int], float] = dict()
        self.existing_runtime_prediction: List[float] = []
        self.queued_query_features: List[np.ndarray] = []
        self.current_time = 0
        self.running_queries: Union[List[str], List[int]] = []
        self.queued_queries: Union[List[str], List[int]] = []
        self.existing_enter_time: List[float] = []
        self.queued_queries_enter_time: List[float] = []
        self.all_query_runtime: MutableMapping[Union[str, int], float] = dict()

    def make_original_prediction(self, trace: pd.DataFrame) -> np.ndarray:
        all_pred, _ = self.predictor.predict(trace, return_per_query=False)
        return all_pred

    def ingest_query(self, start_t: float, query_idx: int):
        return None

    def print_state(self):
        print("current time: ", self.current_time)
        print("running_queries: ", list(zip(self.running_queries, self.existing_runtime_prediction)))
        print("queued_queries: ", self.queued_queries)

    def submit_query(
        self,
        pos_in_queue: int,
        query_rep: Union[str, int],
        pred_runtime: float,
        query_feature: np.ndarray,
        submit_time: float,
        enter_time: float,
        finish_t: float,
        query_concur_features: Optional[torch.Tensor],
        pre_info_length: int,
        new_existing_finish_time: Optional[List[float]] = None,
        new_existing_runtime_prediction: Optional[List[float]] = None,
        new_existing_query_concur_features: Optional[List[Optional[torch.Tensor]]] = None
    ):
        # first upload the prediction on existing runtime when a new query is submitted
        if new_existing_finish_time is not None:
            self.existing_finish_time = new_existing_finish_time
        if new_existing_runtime_prediction is not None:
            self.existing_runtime_prediction = new_existing_runtime_prediction
        if new_existing_query_concur_features is not None:
            self.existing_query_concur_features = new_existing_query_concur_features
        self.running_queries.append(query_rep)
        self.existing_query_features.append(query_feature)
        self.existing_start_time.append(submit_time)
        self.existing_finish_time.append(finish_t)
        self.existing_query_concur_features.append(query_concur_features)
        self.existing_pre_info_length.append(pre_info_length)
        self.existing_enter_time.append(enter_time)
        self.existing_runtime_prediction.append(pred_runtime)
        self.queued_queries.pop(pos_in_queue)
        self.queued_query_features.pop(pos_in_queue)
        self.queued_queries_enter_time.pop(pos_in_queue)


    def finish_query(self, current_time: float = None) -> None:
        if current_time is not None:
            self.current_time = current_time
        pop_index = []
        for i, finish_t in enumerate(self.existing_finish_time):
            if finish_t <= self.current_time:
                pop_index.append(i)
                query_str = self.running_queries[i]
                self.all_query_runtime[query_str] = (
                    finish_t - self.existing_enter_time[i]
                )
        if len(pop_index) == 0:
            return
        length = len(self.existing_finish_time)
        self.running_queries = reverse_index_list(self.running_queries, pop_index)
        self.existing_enter_time = reverse_index_list(self.existing_enter_time, pop_index)
        self.existing_query_features = reverse_index_list(self.existing_query_features, pop_index)
        self.existing_runtime_prediction = reverse_index_list(self.existing_runtime_prediction, pop_index)
        self.existing_start_time = reverse_index_list(self.existing_start_time, pop_index)
        self.existing_finish_time = reverse_index_list(self.existing_finish_time, pop_index)
        # Todo: the last two needs change when we remove a query from its pre info,
        #  or we train with sufficient squence length
        self.existing_query_concur_features = [
            self.existing_query_concur_features[i]
            for i in range(length)
            if i not in pop_index
        ]
        self.existing_pre_info_length = [
            self.existing_pre_info_length[i]
            for i in range(length)
            if i not in pop_index
        ]

    def ingest_query_simulation(
        self,
        start_t: float,
        query_str: Optional[Union[str, int]] = None,
        query_idx: Optional[int] = None,
    ) -> Tuple[bool, bool]:
        """We work on planning the currently queued queries if quert_str is None (i.e., no query submitted)"""
        self.current_time = start_t
        self.finish_query()
        should_immediate_re_ingest = False
        should_pause_and_re_ingest = False
        if query_str is not None:
            self.queued_queries.append(query_str)
            self.queued_queries_enter_time.append(start_t)
            query_feature = self.stage_model.featurize_online(query_idx)
            self.queued_query_features.append(query_feature)

        if len(self.queued_query_features) == 0:
            # nothing to do when there is no query in the queue
            return should_immediate_re_ingest, should_pause_and_re_ingest

        predictions, global_x, global_pre_info_length = self.predictor.online_inference(
            self.existing_query_features,
            self.existing_query_concur_features,
            self.existing_pre_info_length,
            self.queued_query_features,
            self.existing_start_time,
            start_t,
        )

        predictions = predictions.reshape(-1).detach().numpy()
        # Todo: add algorithms to decide whether to put in queue or directly for execution
        if len(self.running_queries) == 0:
            # submit up to self.max_concurrency_level number of queries in queue when there is no query running
            # Todo: this is not optimal
            assert len(predictions) == len(self.queued_queries)
            sort_idx = np.argsort(predictions)
            if len(sort_idx) >= self.max_concurrency_level:
                sort_idx = sort_idx[: self.max_concurrency_level]
            submit_query_str = []
            submit_query_feature = []
            submit_enter_time = []
            submit_pred_runtime = []
            for i in sort_idx:
                submit_query_str.append(self.queued_queries[i])
                submit_query_feature.append(self.queued_query_features[i])
                submit_enter_time.append(self.queued_queries_enter_time[i])
                submit_pred_runtime.append(float(predictions[i]))
            for i, idx in enumerate(sort_idx):
                finish_t = float(predictions[idx]) + start_t
                query_str = submit_query_str[i]
                query_feature = submit_query_feature[i]
                enter_t = submit_enter_time[i]
                pred_runtime = submit_pred_runtime[i]
                self.submit_query(
                    idx,
                    query_str,
                    pred_runtime,
                    query_feature,
                    start_t,
                    enter_t,
                    finish_t,
                    None,
                    int(global_pre_info_length[idx]),
                )
            return should_immediate_re_ingest, should_pause_and_re_ingest
        elif len(self.running_queries) >= self.max_concurrency_level:
            # when the server is running at its full capacity, should pause and retry
            should_pause_and_re_ingest = True
            return should_immediate_re_ingest, should_pause_and_re_ingest
        else:
            # Todo: implement some better algos
            # Todo: add another logic: if the currently queued queries are all "bad" for the system load, pause and retry
            all_new_existing_pred = []
            all_curr_pred = []
            all_delta_sum = []
            all_query_concur_feature = []
            all_global_pre_info_length = []
            all_existing_query_concur_feature = []
            for i in range(len(self.queued_queries)):
                pred_idx = i * (1 + len(self.existing_query_concur_features))
                all_global_pre_info_length.append(global_pre_info_length[pred_idx])
                curr_pred = predictions[pred_idx]
                curr_concur_feature = global_x[pred_idx]
                all_curr_pred.append(curr_pred)
                old_existing_pred = np.asarray(self.existing_runtime_prediction)
                new_existing_pred = predictions[(pred_idx + 1): (pred_idx + len(self.existing_query_concur_features) + 1)]
                curr_existing_query_concur_feature = []
                for j in range(pred_idx + 1, pred_idx + len(self.existing_query_concur_features) + 1):
                    curr_existing_query_concur_feature.append(global_x[j])
                all_new_existing_pred.append(new_existing_pred)
                all_query_concur_feature.append(curr_concur_feature)
                all_existing_query_concur_feature.append(curr_existing_query_concur_feature)
                # realistically, should be a positive number, the smaller, the better
                delta = new_existing_pred - old_existing_pred
                all_delta_sum.append(np.sum(delta))
            # Heuristic to submit the query that incur minimal delta on the existing queries, then resubmit the next
            selected_idx = np.argmin(all_delta_sum)
            finish_t = all_curr_pred[selected_idx] + start_t
            new_existing_finish_time = []
            for i in range(len(self.existing_start_time)):
                new_existing_finish_time.append(all_new_existing_pred[selected_idx][i] + self.existing_start_time[i])
            self.submit_query(
                selected_idx,
                self.queued_queries[selected_idx],
                all_curr_pred[selected_idx],
                self.queued_query_features[selected_idx],
                start_t,
                self.queued_queries_enter_time[selected_idx],
                finish_t,
                all_query_concur_feature[selected_idx],
                int(global_pre_info_length[selected_idx]),
                new_existing_finish_time,
                list(all_new_existing_pred[selected_idx]),
                all_existing_query_concur_feature[selected_idx]
            )
            # immediately resubmit the next
            should_immediate_re_ingest = True
            return should_immediate_re_ingest, should_pause_and_re_ingest

