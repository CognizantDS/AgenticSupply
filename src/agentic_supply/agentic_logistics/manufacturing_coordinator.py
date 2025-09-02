from typing import Any, Dict
from neuro_san.interfaces.coded_tool import CodedTool


from agentic_supply.manufacturing_assistant.scheduling_notifying import get_order_db, Order
from agentic_supply.utilities.config import PRODUCT_NAMES


class ManufacturingScheduler(CodedTool):
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

        product_name: PRODUCT_NAMES = args.get("product_name")
        site_name = args.get("site_name")
        destination = args.get("destination")
        required_delivery_date = args.get("required_delivery_date")
        quantity = args.get("quantity")
        order = Order(
            product_name=product_name,
            destination=destination,
            required_delivery_date=required_delivery_date,
            site_name=site_name,
            quantity=quantity,
        )
        order.schedule()
        completion_duration = order.get_completion_duration()
        return f"The order of {quantity} {product_name} for {site_name} was scheduled with : order_id={order.id}, order_time={order.schedule_time}. The estimated completion_duration is {completion_duration} seconds."


class ManufacturingCompletionVerifier(CodedTool):
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

        order_id = args.get("order_id")
        order_db = get_order_db()
        order = order_db.get_order(order_id)
        status, remaining_time = order.verify_completion_status()
        return f"The order of {order.product_name} in {order.site_name} ({order_id}) is {status} {'' if status == 'complete' else f'({remaining_time} seconds left)'}"
