"""
References :
"""
import uuid
from typing import Dict, List, Tuple, Optional

from agentic_supply.utilities.log_utils import set_logging, get_logger
from agentic_supply.utilities.config import PRODUCT_NAMES


set_logging()
logger = get_logger(__name__)


def order_replenishment(product_name: PRODUCT_NAMES) -> str:
    return uuid.hex

def 


def get_stock_level(product_name: PRODUCT_NAMES) -> float:
    product_to_stock: Dict[PRODUCT_NAMES, float] = {"lactic_acid": 12.3, "ascorbic_acid": 5.7}
    return product_to_stock[product_name]


def get_safety_level(product_name: PRODUCT_NAMES) -> float:
    product_to_safety: Dict[PRODUCT_NAMES, float] = {"lactic_acid": 15.0, "ascorbic_acid": 3.0}
    return product_to_safety[product_name]


def is_replenishment_needed(
    product_name: Optional[PRODUCT_NAMES] = None, stock_level: Optional[float] = None, safety_level: Optional[float] = None
) -> bool:
    if stock_level is None:
        stock_level = get_stock_level(product_name)
    if safety_level is None:
        safety_level = get_safety_level(product_name)
    return stock_level <= safety_level
