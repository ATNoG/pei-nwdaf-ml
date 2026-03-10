import asyncio

from .notifier import AlertLevel, Notifier


class NotificationCenter:

    def __init__(self):
        self.notifiers: list[Notifier] = []

    def register(self, notifier: Notifier):
        self.notifiers.append(notifier)

    async def notify(self, message: str, level: AlertLevel = AlertLevel.INFO):
        await asyncio.gather(*[notifier.notify(message, level) for notifier in self.notifiers])
