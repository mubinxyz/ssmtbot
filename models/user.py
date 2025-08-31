# models/user.py

from sqlalchemy import Column, Integer, String
from services.db_service import Base, get_db
from sqlalchemy.orm import Session

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    
class UnregisteredUser(Base):
    __tablename__ = "unregistered_users"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)


def get_or_create_user(chat_id, username=None, first_name=None, last_name=None):
    """
    Retrieve a user by chat_id, or create if not exists.
    Returns the User instance.
    """
    with get_db() as db:  # get_db yields a SQLAlchemy Session
        user = db.query(User).filter(User.chat_id == chat_id).first()
        if user:
            changed = False
            if username is not None and user.username != username:
                user.username = username
                changed = True
            if first_name is not None and user.first_name != first_name:
                user.first_name = first_name
                changed = True
            if last_name is not None and user.last_name != last_name:
                user.last_name = last_name
                changed = True
            if changed:
                db.add(user)
                db.commit()
                db.refresh(user)
            return user

        user = User(
            chat_id=chat_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


def is_user_registered(chat_id: int) -> bool:
    """
    Check if a user with the given chat_id exists in the database.
    Returns True if exists, False otherwise.
    """
    with get_db() as db:
        return db.query(User).filter(User.chat_id == chat_id).first() is not None
    
    
def save_unregistered_user(chat_id, username=None, first_name=None, last_name=None):
    """
    Save the chat_id and basic info of an unregistered user.
    Can be used later to manually register them.
    """
    with get_db() as db:
        # Check if already saved
        exists = db.query(UnregisteredUser).filter(UnregisteredUser.chat_id == chat_id).first()
        if exists:
            return  # already saved

        unreg = UnregisteredUser(
            chat_id=chat_id,
            username=username,
            first_name=first_name,
            last_name=last_name
        )
        db.add(unreg)
        db.commit()

