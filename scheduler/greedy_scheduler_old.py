import numpy as np
import logging
import copy
from typing import Optional, Tuple, List, Union, MutableMapping
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN
from scheduler.base_scheduler import BaseScheduler


class GreedyScheduler(BaseScheduler):
    def __init__(
        self,
        stage_model: SingleStage,
        predictor: ConcurrentRNN,
        max_concurrency_level: int = 20,
        min_concurrency_level: int = 1,
        starve_penalty: float = 0.5,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        ignore_short_running: bool = False,
        shorting_running_threshold: float = 5.0,
    ):
        """
        :param stage_model: prediction and featurization for a single query
        :param predictor: predict the runtime of concurrent queries
        :param max_concurrency_level: [hyperparameter] the maximal amount of concurrent queries the system can ingest,
                                      can set to a very big value if don't know how to set
        :param min_concurrency_level: [hyperparameter] not useful for greedy scheduler
        :param starve_penalty: Give a penalty for starving a query for too long
        :param debug: set to true to print and log execution info
        :param ignore_short_running: set to true to directly submit short running query to avoid overhead
        :param shorting_running_threshold: consider query with predicted threshold to be shorting running query
        """
        super(GreedyScheduler, self).__init__(
            stage_model,
            predictor,
            max_concurrency_level,
            min_concurrency_level,
            debug=debug,
        )
        self.starve_penalty = starve_penalty
        self.ignore_short_running = ignore_short_running
        self.shorting_running_threshold = shorting_running_threshold
        self.logger = logger

    def ingest_query(
        self,
        start_t: float,
        query_str: Optional[Union[str, int]] = None,
        query_sql: Optional[str] = None,
        query_idx: Optional[int] = None,
        simulation: bool = True,
    ) -> Tuple[bool, bool, Optional[Tuple[Union[str, int], str, int]]]:
        """We work on planning the currently queued queries if query_str is None (i.e., no query submitted)"""
        self.current_time = start_t
        if simulation:
            self.finish_query_simulation()
        else:
            # adjusting the finishing time of running queries (due to error in estimation)
            for i in range(len(self.existing_finish_time)):
                randomness = np.abs(np.random.normal(2, 2))
                self.existing_finish_time[i] = max(
                    self.existing_finish_time[i], self.current_time + randomness
                )
                self.existing_runtime_prediction[i] = self.existing_finish_time[i] - self.existing_enter_time[i]
        should_immediate_re_ingest = False
        should_pause_and_re_ingest = False
        scheduled_submit = None
        if query_str is not None:
            if self.debug and self.logger:
                self.logger.info(
                    f" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
                )
                self.logger.info(f"     Ingesting query {query_str}")
            query_feature = self.stage_model.featurize_online(query_idx)
            if self.ignore_short_running:
                pred = query_feature[0]
                if pred < self.shorting_running_threshold:
                    should_immediate_re_ingest = True
                    scheduled_submit = (query_str, query_sql, query_idx)
                    if self.debug and self.logger:
                        self.logger.info(
                            f"    ||||directly submit {query_str} with predicted average runtime of {pred}"
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
            self.queued_query_features.append(query_feature)

        if len(self.queued_query_features) == 0:
            # nothing to do when there is no query in the queue
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )
        if len(self.existing_finish_time) == 0:
            next_finish_idx = None
            next_finish_time = None
        else:
            next_finish_idx = np.argmin(self.existing_finish_time)
            next_finish_time = self.existing_finish_time[next_finish_idx]

        predictions, global_x, global_pre_info_length = self.predictor.online_inference(
            self.existing_query_features,
            self.existing_query_concur_features,
            self.existing_pre_info_length,
            self.queued_query_features,
            self.existing_start_time,
            start_t,
            next_finish_idx=next_finish_idx,
            next_finish_time=next_finish_time,
            get_next_finish=True,
        )

        predictions = predictions.reshape(-1).detach().numpy()
        if len(self.running_queries) == 0:
            # submit the shortest running query in queue when there is no query running
            # Todo: this is not optimal
            assert len(predictions) == 2 * len(self.queued_queries)
            predictions_query = predictions[0:-1:2]
            selected_idx = np.argmin(predictions_query)
            scheduled_submit = (
                copy.deepcopy(self.queued_queries[selected_idx]),
                copy.deepcopy(self.queued_queries_sql[selected_idx]),
                copy.deepcopy(self.queued_queries_index[selected_idx]),
            )
            self.submit_query(
                selected_idx,
                self.queued_queries[selected_idx],
                predictions_query[selected_idx],
                self.queued_query_features[selected_idx],
                start_t,
                self.queued_queries_enter_time[selected_idx],
                float(predictions_query[selected_idx]) + start_t,
                None,
                int(global_pre_info_length[selected_idx * 2]),
            )
            should_immediate_re_ingest = True
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
                None,
            )
        else:
            all_score = []
            all_query_idx = []
            for i in range(len(self.queued_queries)):
                pred_idx = i * (2 + len(self.existing_query_concur_features))
                curr_pred = predictions[pred_idx]
                submit_after_pred = predictions[pred_idx + 1]
                # how does the predicted runtime of submitting now compare to submitting later
                curr_delta = (
                    curr_pred - submit_after_pred - (next_finish_time - start_t)
                )
                old_existing_pred = np.asarray(self.existing_runtime_prediction)
                new_existing_pred = predictions[
                    (pred_idx + 2) : (
                        pred_idx + len(self.existing_query_concur_features) + 2
                    )
                ]
                # how will this query change the runtime of existing queries in the system ()
                delta = new_existing_pred - old_existing_pred
                delta_sum = np.sum(delta)
                # for every query first judge whether it is good to wait
                # TODO: is there more clever score?
                score = (
                    curr_delta
                    + delta_sum
                    - (start_t - self.queued_queries_enter_time[i])
                    * self.starve_penalty
                )
                if score < 0 or curr_pred < self.shorting_running_threshold:
                    # when the current system state benefit the current query more than
                    # this query's (probably negative) impact on the running queries
                    # more optimal to submit now than later
                    all_score.append(score)
                    all_query_idx.append(i)
                if self.debug and self.logger:
                    self.logger.info(
                        f"    ||||queued query {self.queued_queries[i]} "
                        f"with curr_pred {curr_pred}, curr_delta {curr_delta}, delta_sum {delta_sum}, score {score}"
                    )
            if len(all_score) == 0:
                should_immediate_re_ingest = False
                should_pause_and_re_ingest = False
                scheduled_submit = None
            else:
                # TODO: use linear programming rather than argmax
                best_query_idx = np.argmin(all_score)
                selected_idx = all_query_idx[best_query_idx]
                converted_idx = selected_idx * (
                    2 + len(self.existing_query_concur_features)
                )
                curr_pred_runtime = predictions[converted_idx]
                finish_t = start_t + curr_pred_runtime
                existing_query_concur_features = global_x[converted_idx]
                new_existing_pred = predictions[
                    (converted_idx + 2) : (
                        converted_idx + len(self.existing_query_concur_features) + 2
                    )
                ]
                new_existing_finish_time = []
                for i in range(len(self.existing_start_time)):
                    new_existing_finish_time.append(
                        new_existing_pred[i] + self.existing_start_time[i]
                    )
                new_existing_query_concur_feature = global_x[
                    (converted_idx + 2) : (
                        converted_idx + len(self.existing_query_concur_features) + 2
                    )
                ]
                scheduled_submit = (
                    copy.deepcopy(self.queued_queries[selected_idx]),
                    copy.deepcopy(self.queued_queries_sql[selected_idx]),
                    copy.deepcopy(self.queued_queries_index[selected_idx]),
                )
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
            return (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            )
