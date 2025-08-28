"""
References :
"""

import uuid
import time
from typing import Dict, List, Tuple, Optional

from agentic_supply.utilities.log_utils import set_logging, get_logger
from agentic_supply.utilities.config import PRODUCT_NAMES


set_logging()
logger = get_logger(__name__)


PRODUCT_TO_TIME: Dict[PRODUCT_NAMES, float] = {"lactic_acid": 10.0, "ascorbic_acid": 5.0}


def order_replenishment(product_name: PRODUCT_NAMES):
    return uuid.uuid4().hex, time.time()


def verify_completion_status(product_name: PRODUCT_NAMES, order_id: str, order_time: float) -> Tuple[bool, float]:
    completion_duration = PRODUCT_TO_TIME[product_name]
    elapsed_time = time.time() - order_time
    status = "complete" if elapsed_time >= completion_duration else "incomplete"
    remaining_time = max(0, completion_duration - elapsed_time)
    return (status, remaining_time)
