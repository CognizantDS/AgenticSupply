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


class Product(BaseModel):
    name: str
    safety_level: Optional[float] = Field(default=None)
    stock_level: Optional[float] = Field(default=None)


class ProductsDB(BaseModel):
    products: List[Product]

    def get_product(self, name: str) -> Product:
        return next((elem for elem in self.products if elem.name == name))


class Site(BaseModel):
    name: str
    country: str
    coordinates: List[float]
    products: List[Product]

    def is_replenishment_needed(self, product_name: str) -> bool:
        product = next((elem for elem in self.products if elem.name == product_name))
        product_db = get_products_db()
        product_ = product_db.get_product(product.name)
        return product.stock_level <= product_.safety_level


class SitesDB(BaseModel):
    sites: List[Site]

    def get_site(self, name: str) -> Site:
        return next((elem for elem in self.sites if elem.name == name))

    def get_sites(self, product_name: str) -> List[Site]:
        return [elem for elem in self.sites if product_name in [el.name for el in elem.products]]


def get_products_db():
    json_data = files(data).joinpath("products.json").read_text()
    return ProductsDB.model_validate_json(json_data)


def get_sites_db():
    json_data = files(data).joinpath("sites.json").read_text()
    return SitesDB.model_validate_json(json_data)
