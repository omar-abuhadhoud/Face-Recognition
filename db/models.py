from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config import EMBEDDING_SIZE


class Base(DeclarativeBase):
    pass


class Person(Base):
    """One row per person (one folder in PEOPLE_DIR)."""
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)

    faces: Mapped[list["Face"]] = relationship(back_populates="person", cascade="all, delete-orphan")


class Face(Base):
    """One row per image of a person, with the embedding of the face found in it."""
    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True, nullable=False)
    image_path: Mapped[str] = mapped_column(String, unique=True, nullable=False)   # relative to the program root, e.g. "images/omar/photo.jpg"
    embedding = mapped_column(Vector(EMBEDDING_SIZE), nullable=False)

    person: Mapped[Person] = relationship(back_populates="faces")
