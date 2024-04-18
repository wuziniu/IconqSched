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
        super(GreedyScheduler).__init__(
            stage_model, predictor, max_concurrency_level, min_concurrency_level
        )
