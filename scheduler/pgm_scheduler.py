# This is the baseline implementation of "BI Batch Manager: A System for Managing Batch
# Workloads on Enterprise Data-Warehouses"
import numpy as np
import logging
from typing import Optional, Tuple, List, Union, MutableMapping
from models.single.stage import SingleStage


class PGMScheduler:
    def __init__(
        self,
        stage_model: SingleStage,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        ignore_short_running: bool = False,
        short_running_threshold: float = 5.0,
        use_memory: bool = True,
        admission_threshold: float = 1000,
        consider_top_k: int = 2,
        starve_penalty: float = 0.5,
    ):
        """
        :param stage_model: prediction and featurization for a single query
        :param debug: set to true to print and log execution info
        :param ignore_short_running: set to true to directly submit short running query to avoid overhead
        :param short_running_threshold: consider query with predicted threshold to be shorting running query
        :param use_memory: if true use the estimated memory as threshold to control MPL, else use the estimated runtime
        :param admission_threshold: admit a query if the sum of manipulated variable is below this threshold
        :param consider_top_k: only consider whether to admit the top k queries in the priority queue
        :param starve_penalty: Give a penalty for starving a query for too long
        """
        assert consider_top_k >= 1 and admission_threshold > 0

        self.stage_model = stage_model
        self.running_queries: List[int] = []
        self.running_queries_prediction: List[float] = []
        self.running_queries_enter_time: List[float] = []
        self.running_queries_start_time: List[float] = []
        self.queued_queries: List[int] = []
        self.queued_queries_sql: List[str] = []
        self.queued_queries_index: List[int] = []
        self.queued_queries_prediction: List[float] = []
        self.queued_queries_enter_time: List[float] = []
        self.all_query_runtime: MutableMapping[int, float] = dict()

        self.use_memory = use_memory
        self.ignore_short_running = ignore_short_running
        self.short_running_threshold = short_running_threshold
        self.admission_threshold = admission_threshold
        self.consider_top_k = consider_top_k
        self.starve_penalty = starve_penalty

        self.debug = debug
        self.logger = logger

    def print_state(self):
        if self.logger is None:
            print("current time: ", self.current_time)
            print(
                "running_queries: ",
                self.running_queries,
            )
            print("queued_queries: ", self.queued_queries)
        else:
            self.logger.info(f"current time: {self.current_time}")
            self.logger.info(f"running_queries: {self.running_queries}")
            self.logger.info(f"queued_queries: {self.queued_queries}")

    def submit_query(
        self,
        pos_in_queue: int,
        query_rep: int,
        prediction: float,
        submit_time: float,
        enter_time: float,
    ) -> None:
        self.running_queries.append(query_rep)
        self.running_queries_prediction.append(prediction)
        self.running_queries_enter_time.append(enter_time)
        self.running_queries_start_time.append(submit_time)

        self.queued_queries.pop(pos_in_queue)
        self.queued_queries_sql.pop(pos_in_queue)
        self.queued_queries_index.pop(pos_in_queue)
        self.queued_queries_prediction.pop(pos_in_queue)
        self.queued_queries_enter_time.pop(pos_in_queue)

    def finish_query(self, current_time: float, query_str: int) -> None:
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
        self.running_queries_prediction.pop(finish_idx)
        self.running_queries_enter_time.pop(finish_idx)
        self.running_queries_start_time.pop(finish_idx)

    def ingest_query(
        self,
        start_t: float,
        query_str: Optional[Union[str, int]] = None,
        query_sql: Optional[str] = None,
        query_idx: Optional[int] = None,
        simulation: bool = False,
    ) -> Tuple[bool, bool, Optional[Tuple[Union[str, int], str, int, float]]]:
        """We work on planning the currently queued queries if query_str is None (i.e., no query submitted)"""
        self.current_time = start_t
        should_immediate_re_ingest = False
        should_pause_and_re_ingest = False
        scheduled_submit = None
        if query_str is not None:
            if self.debug:
                if self.logger:
                    self.logger.info(
                        f" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
                    )
                    self.logger.info(f"     Ingesting query {query_str}")
                else:
                    print(f" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& ")
                    print(f"     Ingesting query {query_str}")
            runtime_pred = self.stage_model.cache.online_inference(query_idx, None)
            if self.ignore_short_running:
                if runtime_pred < self.short_running_threshold:
                    should_immediate_re_ingest = True
                    scheduled_submit = (query_str, query_sql, query_idx, 0)
                    if self.debug:
                        if self.logger:
                            self.logger.info(
                                f"    ||||directly submit {query_str} with predicted average runtime of {runtime_pred}"
                            )
                        else:
                            print(
                                f"    ||||directly submit {query_str} with predicted average runtime of {runtime_pred}"
                            )
                    return (
                        should_immediate_re_ingest,
                        should_pause_and_re_ingest,
                        scheduled_submit,
                    )
            self.queued_queries.append(query_str)
            self.queued_queries_sql.append(query_sql)
            self.queued_queries_index.append(query_idx)
            self.queued_queries_enter_time.append(start_t)
            if self.use_memory:
                memory_pred = self.stage_model.memory_est_cache[query_idx]
                self.queued_queries_prediction.append(memory_pred)
            else:
                self.queued_queries_prediction.append(runtime_pred)

        if len(self.queued_queries) == 0:
            # nothing to do when there is no query in the queue
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )
        selected_idx = None
        if len(self.queued_queries) == 1:
            if (
                np.sum(self.running_queries_prediction)
                + self.queued_queries_prediction[0]
                <= self.admission_threshold
            ):
                selected_idx = 0
        else:
            weighted_prediction = [
                (start_t - self.queued_queries_enter_time[i]) * self.starve_penalty
                + self.queued_queries_prediction[i]
                for i in range(len(self.queued_queries))
            ]
            priority_queue = np.argsort(weighted_prediction)[::-1]
            for i, idx in enumerate(priority_queue):
                if i >= self.consider_top_k:
                    break
                curr_pred = self.queued_queries_prediction[idx]
                if (
                    np.sum(self.running_queries_prediction) + curr_pred
                    <= self.admission_threshold
                ):
                    selected_idx = idx
        if selected_idx is not None:
            query_str = self.queued_queries[selected_idx]
            query_sql = self.queued_queries_sql[selected_idx]
            query_idx = self.queued_queries_index[selected_idx]
            queueing_time = start_t - self.queued_queries_enter_time[selected_idx]
            self.submit_query(
                selected_idx,
                query_str,
                prediction=self.queued_queries_prediction[selected_idx],
                submit_time=start_t,
                enter_time=self.queued_queries_enter_time[selected_idx],
            )
            scheduled_submit = (query_str, query_sql, query_idx, queueing_time)
            return (True, False, scheduled_submit)
        else:
            return (False, False, None)
