from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    email = Column(String(255))
    phone = Column(String(255))

    reservations = relationship("Reservation", back_populates="customer")


class Hotel(Base):
    __tablename__ = "hotels"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    location = Column(String(255))
    phone = Column(String(255))

    rooms = relationship("Room", back_populates="hotel")


class RoomType(Base):
    __tablename__ = "room_types"

    id = Column(Integer, primary_key=True)
    type_name = Column(String(255))
    price_per_night = Column(Float)
    max_occupancy = Column(Integer)

    rooms = relationship("Room", back_populates="room_type")


class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True)
    hotel_id = Column(Integer, ForeignKey("hotels.id"))
    room_type_id = Column(Integer, ForeignKey("room_types.id"))
    room_number = Column(String(255))
    status = Column(String(255))

    hotel = relationship("Hotel", back_populates="rooms")
    room_type = relationship("RoomType", back_populates="rooms")
    reservations = relationship("Reservation", back_populates="room")


class Reservation(Base):
    __tablename__ = "reservations"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    room_id = Column(Integer, ForeignKey("rooms.id"))
    check_in = Column(DateTime)
    check_out = Column(DateTime)
    status = Column(String(255))

    customer = relationship("Customer", back_populates="reservations")
    room = relationship("Room", back_populates="reservations")
    payment = relationship("Payment", back_populates="reservation", uselist=False)


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    reservation_id = Column(Integer, ForeignKey("reservations.id"))
    payment_date = Column(DateTime)
    amount = Column(Float)
    payment_method = Column(String(255))

    reservation = relationship("Reservation", back_populates="payment")
