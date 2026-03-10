import logging
import os

import httpx

from .notifier import AlertLevel, Notifier

logger = logging.getLogger(__name__)

LEVEL_COLORS = {
    AlertLevel.INFO: "#36a64f",
    AlertLevel.WARNING: "#ff9900",
    AlertLevel.CRITICAL: "#ff0000",
}


class SlackNotifier(Notifier):

    async def notify(self, message: str, level: AlertLevel = AlertLevel.INFO):
        if os.environ.get("SLACK_NOTIFY", "false").lower() != "true":
            return

        webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        if not webhook_url:
            logger.warning("SLACK_NOTIFY is true but SLACK_WEBHOOK_URL is not set")
            return

        payload = {
            "attachments": [{
                "color": LEVEL_COLORS.get(level, "#36a64f"),
                "text": message,
            }]
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=payload)

        if resp.status_code != 200:
            logger.error(
                "Slack notification failed (%s): %s", resp.status_code, resp.text
            )
