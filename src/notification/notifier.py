"""Base class for notifiers."""

from abc import ABC, abstractmethod

from .alert_level import AlertLevel


class Notifier(ABC):
    @abstractmethod
    async def notify(self, message: str, level: AlertLevel = AlertLevel.INFO):
        pass
