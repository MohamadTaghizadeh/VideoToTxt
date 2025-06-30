from sqlalchemy import Column, String, DateTime, Integer, Enum, SmallInteger, JSON
from dbutils.database import Base
from core.utils import generate_uuid
from dbutils.schemas import WebhookStatus


def str(self) -> str:
    return " , ".join([f"{key}: {value}" for key, value in self.__dict__.items()][::-1])


def items(self) -> dict:
    result = {}
    for k, v in self.__dict__.items():
        if not k.startswith("_"):
            result[k] = v
    return result.items()


Base.__str__ = str
Base.items = items


# Base.__repr__ = str


