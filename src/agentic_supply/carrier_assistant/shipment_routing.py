"""
References :
"""

from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, computed_field
from importlib.resources import files
import uuid
import os
import time

from agentic_supply import data
from agentic_supply.utilities.log_utils import set_logging, get_logger
from agentic_supply.utilities.config import PRODUCT_NAMES, DESTINATIONS, ARTIFACTS_DIR
from agentic_supply.carrier_assistant.transit_querying import LandRoute, OceanRoute

set_logging()
logger = get_logger(__name__)


SHIPMENT_DB_PATH = os.path.join(ARTIFACTS_DIR, "shipment_db.json")


class ShipmentRoute(BaseModel):
    id: str = Field(default=uuid.uuid4().hex)
    land_routes: List[LandRoute]
    ocean_routes: List[OceanRoute]

    @computed_field
    @property
    def total_transit_time(self) -> float:
        transit_times = [elem.transit_time_days for elem in self.land_routes + self.ocean_routes]
        return sum(transit_times)


class Shipment(BaseModel):
    id: str = Field(default=uuid.uuid4().hex)
    shipment_route: ShipmentRoute
    manufacturing_order_id: str
    placed: bool = Field(default=False)
    placement_time: Optional[float] = Field(default=None)

    def place(self):
        self.placed = True
        self.placement_time = time.time()
        shipments_db = get_shipments_db()
        shipments_db.add_shipment(self)


class ShipmentsDB(BaseModel):
    shipments: List[Shipment] = []

    def get_shipment(self, id: str) -> Shipment:
        return next((elem for elem in self.shipments if elem.id == id))

    def save(self):
        with open(SHIPMENT_DB_PATH, "w") as f:
            f.write(self.model_dump_json(indent=4))

    def add_shipment(self, shipment: Shipment):
        self.shipments.append(shipment)
        self.save()


def get_shipments_db():
    with open(SHIPMENT_DB_PATH, "r") as f:
        json_data = f.read()
    return ShipmentsDB.model_validate_json(json_data)
