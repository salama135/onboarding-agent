from dataclasses import dataclass, field

LAPTOP_CATALOG = {
    "MacBook Air": 1200,
    "MacBook Pro 16": 3500,
    "ThinkPad T14": 1100,
    "Dell XPS 15": 2200,
}
ACCOUNT_SYSTEMS = ("email", "slack", "jira", "vpn")


@dataclass
class MockITSystems:
    """In-memory stand-ins for the IT ordering, badge, and account systems."""

    badge_failures_before_success: int = 1
    down_systems: frozenset[str] = frozenset({"vpn"})
    orders: dict[str, dict] = field(default_factory=dict)
    accounts: dict[str, set[str]] = field(default_factory=dict)
    badge_calls: int = 0

    def place_laptop_order(self, employee_id: str, model: str) -> dict:
        if model not in LAPTOP_CATALOG:
            raise ValueError(f"Unknown laptop '{model}'. Choose from: {', '.join(LAPTOP_CATALOG)}.")
        order_id = f"{employee_id}-laptop"  # idempotency key: a retried order never buys a second laptop
        self.orders.setdefault(order_id, {"order_id": order_id, "model": model, "price": LAPTOP_CATALOG[model]})
        return self.orders[order_id]

    def badge_status(self, employee_id: str) -> str:
        self.badge_calls += 1
        if self.badge_calls <= self.badge_failures_before_success:
            raise ConnectionError("Badge system timed out")
        return "ready for pickup at reception"

    def create_account(self, employee_id: str, system: str) -> str:
        if system not in ACCOUNT_SYSTEMS:
            raise ValueError(f"Unknown system '{system}'. Choose from: {', '.join(ACCOUNT_SYSTEMS)}.")
        if system in self.down_systems:
            raise ConnectionError(f"The {system} provisioning service is down")
        self.accounts.setdefault(employee_id, set()).add(system)
        return f"{system} account created"
