from datetime import datetime
from typing import Any

import pytz

KST: Any = pytz.timezone('Asia/Seoul')
DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S.%f%z"

def get_iso_8601_current_time() -> str:
    return datetime.now(KST).strftime(DATE_FORMAT)

def iso_8601_str_to_datetime(iso_8601_time: str) -> datetime:
    return datetime.strptime(iso_8601_time, DATE_FORMAT)

def datetime_to_iso_8601_str(dt: datetime) -> str:
    return dt.strftime(DATE_FORMAT)

def get_kst_now() -> datetime:
    return datetime.now(KST)