"""
References :
"""

import os
import uuid
import time
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

from agentic_supply.utilities.log_utils import set_logging, get_logger
from agentic_supply.utilities.config import PRODUCT_NAMES, DESTINATIONS, ARTIFACTS_DIR
from agentic_supply.inventory_assistant.stock_monitoring import get_products_db


set_logging()
logger = get_logger(__name__)


ORDER_DB_PATH = os.path.join(ARTIFACTS_DIR, "order_db.json")


class Order(BaseModel):
    product_name: str
    site_name: str
    quantity: float
    destination: DESTINATIONS
    required_delivery_date: str
    scheduled: bool = Field(default=False)
    id: Optional[str] = None
    schedule_time: Optional[float] = None

    def schedule(self):
        self.scheduled = True
        self.id = uuid.uuid4().hex
        self.schedule_time = time.time()
        order_db = get_order_db()
        order_db.add_order(self)

    def get_completion_duration(self) -> float:
        products_db = get_products_db()
        product = products_db.get_product(self.product_name)
        return product.production_time_unit * self.quantity

    def verify_completion_status(self) -> Tuple[bool, float]:
        completion_duration = self.get_completion_duration()
        elapsed_time = time.time() - self.schedule_time
        status = "complete" if elapsed_time >= completion_duration else "incomplete"
        remaining_time = max(0, completion_duration - elapsed_time)
        return (status, remaining_time)


class OrderDB(BaseModel):
    orders: List[Order] = []

    def add_order(self, order: Order):
        self.orders.append(order)
        self.save()

    def get_order(self, order_id: str) -> Order:
        return next((elem for elem in self.orders if elem.id == order_id))

    def save(self):
        with open(ORDER_DB_PATH, "w") as f:
            f.write(self.model_dump_json(indent=4))


def get_order_db():
    with open(ORDER_DB_PATH, "r") as f:
        json_data = f.read()
    return OrderDB.model_validate_json(json_data)
