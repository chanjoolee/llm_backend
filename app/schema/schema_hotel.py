from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class CustomerSchema(BaseModel):
    id: int
    name: str
    email: str
    phone: str

    class Config:
        orm_mode = True


class HotelSchema(BaseModel):
    id: int
    name: str
    location: str
    phone: str

    class Config:
        orm_mode = True


class RoomTypeSchema(BaseModel):
    id: int
    type_name: str
    price_per_night: float
    max_occupancy: int

    class Config:
        orm_mode = True


class RoomSchema(BaseModel):
    id: int
    hotel_id: int
    room_type_id: int
    room_number: str
    status: str

    class Config:
        orm_mode = True


class ReservationSchema(BaseModel):
    id: int
    customer_id: int
    room_id: int
    check_in: datetime
    check_out: datetime
    status: str

    class Config:
        orm_mode = True


class PaymentSchema(BaseModel):
    id: int
    reservation_id: int
    payment_date: datetime
    amount: float
    payment_method: str

    class Config:
        orm_mode = True
