from datetime import datetime


def timestamp():
    datetime.now().strftime('%H:%M:%S')
