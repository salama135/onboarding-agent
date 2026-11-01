from typing import Literal

from langchain.tools import ToolRuntime, tool

from onboarding.context import OnboardingContext
from onboarding.mock_systems import LAPTOP_CATALOG

Runtime = ToolRuntime[OnboardingContext]


@tool
def list_laptops() -> str:
    """List the laptop models IT can order, with prices."""
    return "; ".join(f"{name} (${price})" for name, price in LAPTOP_CATALOG.items())


@tool
def order_laptop(model: str, runtime: Runtime) -> str:
    """Order a laptop for the employee. A manager must approve before it runs."""
    try:
        order = runtime.context.it_systems.place_laptop_order(runtime.context.user_id, model)
    except ValueError as err:
        return str(err)
    return f"Order {order['order_id']} placed: {order['model']} (${order['price']}). Delivery in 3 days."


@tool
def check_badge_status(runtime: Runtime) -> str:
    """Check whether the employee's office badge is ready."""
    return f"Badge is {runtime.context.it_systems.badge_status(runtime.context.user_id)}."


@tool
def create_account(system: Literal["email", "slack", "jira", "vpn"], runtime: Runtime) -> str:
    """Create one IT account for the employee. Call once per system."""
    return runtime.context.it_systems.create_account(runtime.context.user_id, system)


IT_TOOLS = [list_laptops, order_laptop, check_badge_status, create_account]
