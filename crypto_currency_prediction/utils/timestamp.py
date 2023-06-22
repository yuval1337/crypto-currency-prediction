from datetime import datetime


def timestamp() -> str:
  return datetime.now().strftime('%d%m%y%H%M%S')
