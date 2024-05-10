import numpy as np
from typing import Optional, Tuple, List, Union, MutableMapping
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN
from scheduler.base_scheduler import BaseScheduler


class GreedyScheduler(BaseScheduler):
    def __init__(
        self,
        stage_model: SingleStage,
        predictor: ConcurrentRNN,
        max_concurrency_level: int = 10,
        min_concurrency_level: int = 2,
    ):
        super(GreedyScheduler, self).__init__(
            stage_model, predictor, max_concurrency_level, min_concurrency_level
        )

    def ingest_query_simulation(
        self,
        start_t: float,
        query_str: Optional[Union[str, int]] = None,
        query_idx: Optional[int] = None,
    ) -> Tuple[bool, bool, Optional[float]]:
        """We work on planning the currently queued queries if query_str is None (i.e., no query submitted)"""
        self.current_time = start_t
        self.finish_query()
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
        # Todo: add algorithms to decide whether to put in queue or directly for execution
        if len(self.running_queries) == 0:
            # submit the shortest running query in queue when there is no query running
            # Todo: this is not optimal
            assert len(predictions) == 2 * len(self.queued_queries)

            predictions_query = predictions[0:-1:2]
            selected_idx = np.argmin(predictions_query)
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
                scheduled_submit,
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
                    curr_pred - submit_after_pred + (next_finish_time - start_t)
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
                if curr_delta + delta_sum < 0:
                    # when the current system state benefit the current query more than
                    # this query's (probably negative) impact on the running queries
                    # more optimal to submit now than later
                    all_score.append(delta_sum + curr_delta)
                    all_query_idx.append(i)
                    # todo: add more clever conditions
            if len(all_score) == 0:
                should_immediate_re_ingest = False
                should_pause_and_re_ingest = False
                # Todo implement scheduled submit in the future
                # now we just pause and wait for re_ingest
                scheduled_submit = None
            else:
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
