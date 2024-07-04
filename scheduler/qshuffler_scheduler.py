# This is the baseline implementation of "Interaction-aware scheduling of report-generation workloads"
import numpy as np
import logging
from typing import Optional, Tuple, List, Union, MutableMapping
from models.single.stage import SingleStage
from models.concurrency.baselines.qshuffler_estimator import QEstimator


class QShuffler:
    def __init__(
        self,
        stage_model: SingleStage,
        cost_model: QEstimator,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        ignore_short_running: bool = False,
        short_running_threshold: float = 5.0,
        lookahead: int = 10,
        cost_threshold: Optional[float] = None,
        mpl: int = 5,
    ):
        """
        :param stage_model: prediction and featurization for a single query
        :param cost_model: predicting the runtime of query mix
        :param debug: set to true to print and log execution info
        :param ignore_short_running: set to true to directly submit short running query to avoid overhead
        :param short_running_threshold: consider query with predicted threshold to be shorting running query
        :param cost_threshold: theta_NRO in the original paper
        :param mpl: The multi-programing level, will not submit query if number of concurrent query >= MPL
        :param lookahead: The size of the queue. If there are more queries in the queue than lookahead, the algorithm
                          will force to submit query regardless of MPL
        """
        assert mpl >= 1 and cost_threshold > 0

        self.stage_model = stage_model
        self.cost_model = cost_model
        self.running_queries: List[int] = []
        self.running_queries_type: List[float] = []
        self.running_queries_enter_time: List[float] = []
        self.running_queries_start_time: List[float] = []
        self.running_queries_feature = np.zeros(self.cost_model.num_clusters)
        self.queued_queries: List[int] = []
        self.queued_queries_sql: List[str] = []
        self.queued_queries_index: List[int] = []
        self.queued_queries_type: List[int] = []
        self.queued_queries_enter_time: List[float] = []
        self.all_query_runtime: MutableMapping[int, float] = dict()

        self.lookahead = lookahead
        self.ignore_short_running = ignore_short_running
        self.short_running_threshold = short_running_threshold
        if cost_threshold is not None:
            self.cost_threshold = cost_threshold
        else:
            self.cost_threshold = self.cost_model.cost_threshold
        self.mpl = mpl

        self.debug = debug
        self.logger = logger

    def submit_query(
        self,
        pos_in_queue: int,
        query_rep: int,
        query_type: int,
        submit_time: float,
        enter_time: float,
    ) -> None:
        self.running_queries.append(query_rep)
        self.running_queries_type.append(query_type)
        self.running_queries_enter_time.append(enter_time)
        self.running_queries_start_time.append(submit_time)
        self.running_queries_feature[query_type] += 1

        self.queued_queries.pop(pos_in_queue)
        self.queued_queries_sql.pop(pos_in_queue)
        self.queued_queries_index.pop(pos_in_queue)
        self.queued_queries_type.pop(pos_in_queue)
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
        query_type = self.running_queries_type.pop(finish_idx)
        self.running_queries_enter_time.pop(finish_idx)
        self.running_queries_start_time.pop(finish_idx)
        assert self.running_queries_feature[query_type] > 0
        self.running_queries_feature[query_type] -= 1

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
            query_type = self.cost_model.get_query_type(query_idx)
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
            self.queued_queries_type.append(query_type)
            self.queued_queries_enter_time.append(start_t)

        if len(self.queued_queries) == 0:
            # nothing to do when there is no query in the queue
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )
        selected_idx = None
        if len(self.queued_queries) == 1:
            if len(self.running_queries) < self.mpl:
                selected_idx = 0
        else:
            if (
                len(self.running_queries) >= self.mpl
                and len(self.queued_queries) < self.lookahead
            ):
                selected_idx = None
            else:
                scores = self.cost_model.online_inference(self.running_queries_feature)
                priority = 1 / np.abs(self.cost_threshold - scores) + 1e-3
                priority_queue = np.argsort(priority)[::-1]
                for idx in priority_queue:
                    if idx in self.queued_queries_type:
                        selected_idx = self.queued_queries_type.index(idx)
                        break
        if selected_idx is not None:
            query_str = self.queued_queries[selected_idx]
            query_sql = self.queued_queries_sql[selected_idx]
            query_idx = self.queued_queries_index[selected_idx]
            queueing_time = start_t - self.queued_queries_enter_time[selected_idx]
            self.submit_query(
                selected_idx,
                query_str,
                query_type=self.queued_queries_type[selected_idx],
                submit_time=start_t,
                enter_time=self.queued_queries_enter_time[selected_idx],
            )
            scheduled_submit = (query_str, query_sql, query_idx, queueing_time)
            return (True, False, scheduled_submit)
        else:
            return (False, False, None)
