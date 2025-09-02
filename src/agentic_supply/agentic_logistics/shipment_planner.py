from typing import Any, Dict
from neuro_san.interfaces.coded_tool import CodedTool


from agentic_supply.carrier_assistant.shipment_routing import get_shipments_db, Shipment, ShipmentRoute
from agentic_supply.carrier_assistant.transit_querying import get_land_routes_db, get_ocean_routes_db
from agentic_supply.utilities.config import PRODUCT_NAMES


class ShipmentPlanner(CodedTool):
    """
    CodedTool implementation of a calculator for the math_guy test.

    Upon activation by the agent hierarchy, a CodedTool will have its
    invoke() call called by the system.

    Implementations are expected to clean up after themselves.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """
        Called when the coded tool is invoked asynchronously by the agent hierarchy.
        Strongly consider overriding this method instead of the "easier" synchronous
        version above when the possibility of making any kind of call that could block
        (like sleep() or a socket read/write out to a web service) is within the
        scope of your CodedTool.

        :param args: An argument dictionary whose keys are the parameters
                to the coded tool and whose values are the values passed for them
                by the calling agent.  This dictionary is to be treated as read-only.
        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.

                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.
        :return: A return value that goes into the chat stream.
        """
        manufacturing_order_id = args.get("manufacturing_order_id")
        land_routes_ids: str = args.get("land_routes_ids")
        ocean_routes_ids: str = args.get("ocean_routes_ids")

        land_routes_db = get_land_routes_db()
        ocean_routes_db = get_ocean_routes_db()
        land_routes = [land_routes_db.get_route(id=elem.strip()) for elem in land_routes_ids.split(",")]
        ocean_routes = [ocean_routes_db.get_route(id=elem.strip()) for elem in ocean_routes_ids.split(",")]

        shipment = Shipment(
            manufacturing_order_id=manufacturing_order_id, shipment_route=ShipmentRoute(land_routes=land_routes, ocean_routes=ocean_routes)
        )
        shipment.place()
        return str(shipment)


class ShipmentQuerier(CodedTool):
    """
    CodedTool implementation of a calculator for the math_guy test.

    Upon activation by the agent hierarchy, a CodedTool will have its
    invoke() call called by the system.

    Implementations are expected to clean up after themselves.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """
        Called when the coded tool is invoked asynchronously by the agent hierarchy.
        Strongly consider overriding this method instead of the "easier" synchronous
        version above when the possibility of making any kind of call that could block
        (like sleep() or a socket read/write out to a web service) is within the
        scope of your CodedTool.

        :param args: An argument dictionary whose keys are the parameters
                to the coded tool and whose values are the values passed for them
                by the calling agent.  This dictionary is to be treated as read-only.
        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.

                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.
        :return: A return value that goes into the chat stream.
        """
        shipments_db = get_shipments_db()
        return shipments_db.model_dump_json()
