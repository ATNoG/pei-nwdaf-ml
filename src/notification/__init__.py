from .alert_level import AlertLevel
from .center import NotificationCenter
from .slack import SlackNotifier

notification_center = NotificationCenter()
notification_center.register(SlackNotifier())
# register other notifiers here
