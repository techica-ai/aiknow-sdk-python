"""Utility tools: format_currency, current_datetime."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiknow.agents.builtin_tools._base import BuiltinTool, ToolParam

if TYPE_CHECKING:
    from aiknow.agents.builtin_tools._toolbox import BuiltinToolContext


class FormatCurrencyTool(BuiltinTool):
    """Format an amount as a localized currency string."""

    name = "format_currency"
    description = {
        "vi": "Định dạng số tiền theo đơn vị tiền tệ",
        "en": "Format an amount as a localized currency string",
    }
    parameters = [
        ToolParam(
            name="amount",
            type="number",
            description={"en": "Numeric amount to format"},
            required=True,
        ),
        ToolParam(
            name="currency",
            type="string",
            description={"en": "Currency code (e.g. VND, USD, EUR)"},
            required=False,
            default="VND",
        ),
        ToolParam(
            name="locale",
            type="string",
            description={"en": "Locale code for formatting (e.g. vi-VN, en-US)"},
            required=False,
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        amount = kwargs.get("amount", 0)
        currency: str = kwargs.get("currency", "VND").upper()

        try:
            amount = float(amount)
        except (ValueError, TypeError):
            return f"[format_currency] Invalid amount: {amount}"

        # Simple locale-aware formatting
        if currency == "VND":
            # Vietnamese dong — no decimal places
            formatted = f"{amount:,.0f}".replace(",", ".")
            return f"{formatted} ₫"
        elif currency == "USD":
            return f"${amount:,.2f}"
        elif currency == "EUR":
            return f"€{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"


class CurrentDatetimeTool(BuiltinTool):
    """Get the current date and time in the agent's locale timezone."""

    name = "current_datetime"
    description = {
        "vi": "Lấy ngày giờ hiện tại",
        "en": "Get the current date and time",
    }
    parameters = [
        ToolParam(
            name="format",
            type="string",
            description={"en": "Output format: 'full', 'date', 'time', or a strftime pattern"},
            required=False,
            default="full",
        ),
        ToolParam(
            name="timezone",
            type="string",
            description={
                "en": "Timezone name (e.g. 'Asia/Ho_Chi_Minh'). Uses system TZ if omitted."
            },
            required=False,
            default="Asia/Ho_Chi_Minh",
        ),
    ]

    async def execute(self, ctx: BuiltinToolContext, **kwargs: Any) -> str:
        fmt: str = kwargs.get("format", "full")
        tz_name: str = kwargs.get("timezone", "Asia/Ho_Chi_Minh")

        try:
            import zoneinfo  # noqa: PLC0415
            from datetime import datetime  # noqa: PLC0415

            try:
                tz = zoneinfo.ZoneInfo(tz_name)
            except Exception:
                tz = zoneinfo.ZoneInfo("UTC")

            now = datetime.now(tz)

            if fmt == "full":
                return now.strftime("%Y-%m-%d %H:%M:%S %Z")
            elif fmt == "date":
                return now.strftime("%Y-%m-%d")
            elif fmt == "time":
                return now.strftime("%H:%M:%S %Z")
            else:
                try:
                    return now.strftime(fmt)
                except Exception:
                    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

        except Exception:  # noqa: BLE001
            from datetime import datetime  # noqa: PLC0415
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
