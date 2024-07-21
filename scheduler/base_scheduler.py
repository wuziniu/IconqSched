import copy
import logging
import pandas as pd
import numpy as np
import torch
from typing import Optional, Tuple, List, Union, MutableMapping
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN


def reverse_index_list(lst: List, pop_index: List[int]) -> List:
    return [lst[i] for i in range(len(lst)) if i not in pop_index]


class BaseScheduler:
    def __init__(
        self,
        stage_model: SingleStage,
        predictor: Optional[ConcurrentRNN],
        max_concurrency_level: int = 10,
        min_concurrency_level: int = 2,
        future_time_interval: float = 5.0,
        num_time_interval: int = 1,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        ignore_short_running: bool = False,
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
        self.future_time_interval = future_time_interval
        self.num_time_interval = num_time_interval

        self.existing_query_features: List[np.ndarray] = []
        self.existing_query_concur_features: List[Optional[torch.Tensor]] = []
        self.existing_pre_info_length: List[int] = []
        self.existing_start_time: List[float] = []
        self.existing_finish_time: List[float] = []
        self.existing_runtime_prediction_dict: MutableMapping[
            Union[str, int], float
        ] = dict()
        self.existing_runtime_prediction: List[float] = []
        self.existing_runtime_prediction_adjusted: List[float] = []
        self.existing_enter_time: List[float] = []

        self.current_time = 0
        self.running_queries: Union[List[str], List[int]] = []
        self.queued_queries: Union[List[str], List[int]] = []
        self.queued_queries_sql: List[str] = []
        self.queued_queries_index: List[int] = []
        self.queued_query_features: List[np.ndarray] = []
        self.queued_queries_enter_time: List[float] = []
        self.all_query_runtime: MutableMapping[Union[str, int], float] = dict()
        self.debug = debug
        self.logger = logger
        self.ignore_short_running = ignore_short_running

    def make_original_prediction(self, trace: pd.DataFrame) -> np.ndarray:
        all_pred, _ = self.predictor.predict(trace, return_per_query=False)
        return all_pred

    def ingest_query(
        self,
        start_t: float,
        query_str: Optional[Union[str, int]] = None,
        query_sql: Optional[str] = None,
        query_idx: Optional[int] = None,
        simulation: bool = False,
    ):
        return None

    def print_state(self):
        if self.logger is None:
            print("current time: ", self.current_time)
            print(
                "running_queries: ",
                list(zip(self.running_queries, self.existing_runtime_prediction)),
            )
            print("queued_queries: ", self.queued_queries)
        else:
            self.logger.info(f"current time: {self.current_time}")
            self.logger.info(
                f"running_queries: {list(zip(self.running_queries, self.existing_runtime_prediction))}"
            )
            self.logger.info(f"queued_queries: {self.queued_queries}")

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
        new_existing_query_concur_features: Optional[
            List[Optional[torch.Tensor]]
        ] = None,
    ):
        # first upload the prediction on existing runtime when a new query is submitted
        if new_existing_finish_time is not None:
            self.existing_finish_time = new_existing_finish_time
        if new_existing_runtime_prediction is not None:
            self.existing_runtime_prediction = new_existing_runtime_prediction
            self.existing_runtime_prediction_adjusted = copy.deepcopy(
                new_existing_runtime_prediction
            )
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
        self.existing_runtime_prediction_adjusted.append(pred_runtime)
        self.queued_queries.pop(pos_in_queue)
        self.queued_queries_sql.pop(pos_in_queue)
        self.queued_queries_index.pop(pos_in_queue)
        self.queued_query_features.pop(pos_in_queue)
        self.queued_queries_enter_time.pop(pos_in_queue)
        if self.debug and self.logger:
            self.logger.info(
                f"*****submit query {query_rep} with prediction {pred_runtime}******"
            )

    def finish_query(self, current_time: float, query_str: Union[str, int]) -> None:
        self.current_time = current_time
        if query_str not in self.running_queries:
            if not self.ignore_short_running:
                print(f"!!!!!!!!!!!!!Warning: {query_str} is already finished")
                if self.logger is not None:
                    self.logger.warning(
                        f"!!!!!!!!!!!!!Warning: {query_str} is already finished"
                    )
            return
        finish_idx = self.running_queries.index(query_str)
        self.running_queries.pop(finish_idx)
        self.existing_enter_time.pop(finish_idx)
        self.existing_query_features.pop(finish_idx)
        popped_pred_runtime = self.existing_runtime_prediction.pop(finish_idx)
        _ = self.existing_runtime_prediction_adjusted.pop(finish_idx)
        self.existing_start_time.pop(finish_idx)
        self.existing_finish_time.pop(finish_idx)
        # Todo: the last two needs change when we remove a query from its pre info,
        #  or we train with sufficient squence length
        self.existing_query_concur_features.pop(finish_idx)
        self.existing_pre_info_length.pop(finish_idx)
        # adjusting the finishing time of running queries (due to error in estimation)
        for i in range(len(self.existing_finish_time)):
            randomness = np.abs(np.random.normal(2, 1))
            self.existing_finish_time[i] = max(
                self.existing_finish_time[i], current_time + randomness
            )
        if self.debug and self.logger:
            self.logger.info(
                f"*****query {query_str} finished with prediction {popped_pred_runtime}******"
            )

    def finish_query_simulation(self, current_time: float = None) -> None:
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
        self.existing_enter_time = reverse_index_list(
            self.existing_enter_time, pop_index
        )
        self.existing_query_features = reverse_index_list(
            self.existing_query_features, pop_index
        )
        self.existing_runtime_prediction = reverse_index_list(
            self.existing_runtime_prediction, pop_index
        )
        self.existing_runtime_prediction_adjusted = reverse_index_list(
            self.existing_runtime_prediction_adjusted, pop_index
        )
        self.existing_start_time = reverse_index_list(
            self.existing_start_time, pop_index
        )
        self.existing_finish_time = reverse_index_list(
            self.existing_finish_time, pop_index
        )
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

    def compute_score(
        self, predictions: np.ndarray, future_t: float = 0
    ) -> List[float]:
        all_delta_sum = []
        for i in range(len(self.queued_queries)):
            pred_idx = i * (1 + len(self.existing_query_concur_features))
            curr_pred = predictions[pred_idx] + future_t
            curr_delta = curr_pred - self.queued_query_features[i][0]
            old_existing_pred = np.asarray(self.existing_runtime_prediction)
            new_existing_pred = predictions[
                (pred_idx + 1) : (
                    pred_idx + len(self.existing_query_concur_features) + 1
                )
            ]
            delta = new_existing_pred - old_existing_pred
            all_delta_sum.append(np.sum(delta) + curr_delta)
        return all_delta_sum

    def ingest_query_simulation(
        self,
        start_t: float,
        query_str: Optional[Union[str, int]] = None,
        query_idx: Optional[int] = None,
    ) -> Tuple[bool, bool, Optional[float]]:
        """We work on planning the currently queued queries if quert_str is None (i.e., no query submitted)"""
        self.current_time = start_t
        self.finish_query_simulation()
        should_immediate_re_ingest = False
        should_pause_and_re_ingest = False
        scheduled_submit = None
        if query_str is not None:
            self.queued_queries.append(query_str)
            self.queued_queries_enter_time.append(start_t)
            query_feature = self.stage_model.featurize_online(query_idx)
            self.queued_query_features.append(query_feature)

        if len(self.queued_query_features) == 0:
            # nothing to do when there is no query in the queue
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )

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
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )
        elif len(self.running_queries) >= self.max_concurrency_level:
            # when the system is overloaded, should pause and retry
            should_pause_and_re_ingest = True
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )
        elif len(self.running_queries) <= self.min_concurrency_level:
            # when the system is underloaded, should directly submit the "optimal" query
            # Todo: implement some better algos to determine the "optimal" query
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
                # delta between running with current load and average load
                curr_delta = curr_pred - self.queued_query_features[i][0]
                curr_concur_feature = global_x[pred_idx]
                all_curr_pred.append(curr_pred)
                old_existing_pred = np.asarray(self.existing_runtime_prediction)
                new_existing_pred = predictions[
                    (pred_idx + 1) : (
                        pred_idx + len(self.existing_query_concur_features) + 1
                    )
                ]
                curr_existing_query_concur_feature = []
                for j in range(
                    pred_idx + 1,
                    pred_idx + len(self.existing_query_concur_features) + 1,
                ):
                    curr_existing_query_concur_feature.append(global_x[j])
                all_new_existing_pred.append(new_existing_pred)
                all_query_concur_feature.append(curr_concur_feature)
                all_existing_query_concur_feature.append(
                    curr_existing_query_concur_feature
                )
                # realistically, should be a positive number, the smaller, the better
                delta = new_existing_pred - old_existing_pred
                all_delta_sum.append(np.sum(delta) + curr_delta)
            # Heuristic to submit the query that incur minimal delta on the existing queries, then resubmit the next
            selected_idx = np.argmin(all_delta_sum)
            finish_t = all_curr_pred[selected_idx] + start_t
            new_existing_finish_time = []
            for i in range(len(self.existing_start_time)):
                new_existing_finish_time.append(
                    all_new_existing_pred[selected_idx][i] + self.existing_start_time[i]
                )
            self.submit_query(
                selected_idx,
                self.queued_queries[selected_idx],
                all_curr_pred[selected_idx],
                self.queued_query_features[selected_idx],
                start_t,
                self.queued_queries_enter_time[selected_idx],
                finish_t,
                all_query_concur_feature[selected_idx],
                int(all_global_pre_info_length[selected_idx]),
                new_existing_finish_time,
                list(all_new_existing_pred[selected_idx]),
                all_existing_query_concur_feature[selected_idx],
            )
            # immediately resubmit the next
            should_immediate_re_ingest = True
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )
        else:
            # when system is not overloaded/underloaded, decide what and when to submit the "optimal" query
            # TODO: implement some better algos to determine the "optimal" query and time
            # the "when" part somehow doesn't work
            future_submit_time_all = [
                self.future_time_interval * (i + 1)
                for i in range(self.num_time_interval)
            ]
            all_score = self.compute_score(predictions, 0)
            time_feature_pos = len(self.queued_query_features[0]) * 2 + 2
            for future_submit_time in future_submit_time_all:
                future_x = change_time_feature(
                    global_x,
                    len(self.existing_query_concur_features),
                    future_submit_time,
                    time_feature_pos,
                )
                future_predictions = self.predictor.model(
                    future_x, None, global_pre_info_length, False
                )
                future_predictions = future_predictions.reshape(-1).detach().numpy()
                future_all_score = self.compute_score(
                    future_predictions, future_submit_time
                )
                all_score.extend(future_all_score)
            best_query = np.argmin(all_score)
            if best_query >= len(self.queued_queries):
                should_immediate_re_ingest = False
                should_pause_and_re_ingest = True
                # Todo implement scheduled submit in the future
                # now we just pause wait for re_ingest
                scheduled_submit = None
            else:
                selected_idx = best_query
                converted_idx = selected_idx * (
                    1 + len(self.existing_query_concur_features)
                )
                curr_pred_runtime = predictions[converted_idx]
                finish_t = start_t + curr_pred_runtime
                existing_query_concur_features = global_x[converted_idx]
                new_existing_pred = predictions[
                    (converted_idx + 1) : (
                        converted_idx + len(self.existing_query_concur_features) + 1
                    )
                ]
                new_existing_finish_time = []
                for i in range(len(self.existing_start_time)):
                    new_existing_finish_time.append(
                        new_existing_pred[i] + self.existing_start_time[i]
                    )
                new_existing_query_concur_feature = global_x[
                    (converted_idx + 1) : (
                        converted_idx + len(self.existing_query_concur_features) + 1
                    )
                ]
                self.submit_query(
                    selected_idx,
                    self.queued_queries[selected_idx],
                    curr_pred_runtime,
                    self.queued_query_features[selected_idx],
                    start_t,
                    self.queued_queries_enter_time[selected_idx],
                    finish_t,
                    existing_query_concur_features,
                    int(global_pre_info_length[converted_idx]),
                    new_existing_finish_time,
                    list(new_existing_pred),
                    new_existing_query_concur_feature,
                )
                should_immediate_re_ingest = True
                should_pause_and_re_ingest = False
                scheduled_submit = None
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )


def change_time_feature(
    old_feature: List[torch.Tensor],
    existing_query_len: int,
    t: float,
    time_feature_pos: int,
) -> List[torch.Tensor]:
    new_feature = copy.deepcopy(old_feature)
    for i in range(len(new_feature)):
        if i % existing_query_len == 0:
            # this is the queued query
            for j in range(len(new_feature[i])):
                new_feature[i][j, time_feature_pos] = (
                    new_feature[i][j, time_feature_pos] - t
                )
        else:
            # this is the running query
            # TODO: should probably be different in the future
            for j in range(len(new_feature[i])):
                new_feature[i][j, time_feature_pos] = (
                    new_feature[i][j, time_feature_pos] - t
                )
    return new_feature
