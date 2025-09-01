"""
References :
"""

from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
from importlib.resources import files

from agentic_supply import data
from agentic_supply.utilities.log_utils import set_logging, get_logger


set_logging()
logger = get_logger(__name__)


class Port(BaseModel):
    port_name: str
    country: str
    un_locode: str
    coordinates: List[float]
    timezone: str
    handling_time_days: int


class PortsDB(BaseModel):
    ports: List[Port] = []

    def get_port(self, port_name: str) -> Port:
        return next((elem for elem in self.ports if elem.port_name == port_name))


def get_port_db():
    json_data = files(data).joinpath("ports.json").read_text()
    return PortsDB.model_validate_json(json_data)


class OceanRoute(BaseModel):
    origin: str
    destination: str
    carrier: str
    scenario: str
    transit_time_days: int
    cost_usd: int
    frequency_per_week: int
    via: List[str]


class OceanRoutesDB(BaseModel):
    ocean_routes: List[OceanRoute] = []

    def get_routes(self, origin: str, destination: str) -> Port:
        return [elem for elem in self.ocean_routes if elem.origin == origin and elem.destination == destination]


def get_ocean_routes_db():
    json_data = files(data).joinpath("ocean_routes.json").read_text()
    return OceanRoutesDB.model_validate_json(json_data)


class LandRoute(BaseModel):
    origin: str
    destination: str
    carrier: str
    transport_mode: str
    scenario: str
    transit_time_days: float
    cost_usd: float
    distance_km: float


class LandRoutesDB(BaseModel):
    land_routes: List[LandRoute] = []

    def get_routes(self, origin: str, destination: str, transport_mode: Optional[str] = None) -> Port:
        return [
            elem
            for elem in self.land_routes
            if elem.origin == origin
            and elem.destination == destination
            and (transport_mode is None or elem.transport_mode == transport_mode)
        ]


def get_land_routes_db():
    json_data = files(data).joinpath("land_routes.json").read_text()
    return LandRoutesDB.model_validate_json(json_data)
