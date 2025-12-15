"""Class designed to monitor performance of default models"""

import logging
from typing_extensions import Callable

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    inferences:dict[str,dict[tuple[int,int],float]] = {} # inferences stored in runtime per cell

    @classmethod
    def store(cls,analytics_type:str,predicted_value:float, cell_index:int, window_size:int) -> None:
        """Saves predicted value internally"""
        if analytics_type not in cls.inferences:
            cls.inferences[analytics_type] = {}

        storage = cls.inferences[analytics_type]

        # create entries if needed
        tup:tuple[int,int] = (cell_index,window_size)
        # store value for later
        storage[tup] = predicted_value

    @classmethod
    def eval(cls,analytics_type , true_value:float, cell_index:int,window_size:int, callback_function:Callable|None = None):
        """Evaluates loss based on true_value. TS"""

        if analytics_type not in cls.inferences:
            logger.warning(f"No record for analytics [{analytics_type}] found")
            return

        storage = cls.inferences[analytics_type]

        tup:tuple[int,int] = (cell_index,window_size)
        if tup not in storage:
            logger.warning(f"No stored information for {cell_index} for {window_size}s windows")
            return

        predicted_value = storage[tup]
        if predicted_value is None:
            logger.warning(f"No predicted value for ({cell_index}, {window_size}s)")
            return

        # calculate mse
        mse = (predicted_value - true_value) ** 2

        if callback_function is not None:
            logger.info("Evaluation done. starting callback")
            callback_function(window_size,cell_index,mse)
