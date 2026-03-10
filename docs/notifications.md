# Notifications

The ML service sends alerts via a pluggable notification system when:
- **Model performance degrades** (CRITICAL - red)
- **Best model changes** after retraining (INFO - green)

## Setup

### Slack

1. Create a Slack App at https://api.slack.com/apps
2. Enable **Incoming Webhooks** and add one to your channel
3. Add to `.env`:
   ```
   SLACK_NOTIFY=true
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   ```

## Creating a new notifier

1. Create a new file in `src/notification/`, e.g. `discord.py`
2. Extend `Notifier` and implement `notify()`:

```python
from .notifier import Notifier
from .alert_level import AlertLevel


class DiscordNotifier(Notifier):

    async def notify(self, message: str, level: AlertLevel = AlertLevel.INFO):
        # your implementation here
        ...
```

3. Register it in `src/notification/__init__.py`:

```python
from .discord import DiscordNotifier

notification_center.register(DiscordNotifier())
```

## Alert levels

| Level | Use case | Slack color |
|-------|----------|-------------|
| `AlertLevel.INFO` | Model changes, status updates | Green |
| `AlertLevel.WARNING` | Non-critical issues | Orange |
| `AlertLevel.CRITICAL` | Performance degradation | Red |

## Usage

```python
from src.notification import notification_center, AlertLevel

await notification_center.notify("something happened", AlertLevel.WARNING)
```
