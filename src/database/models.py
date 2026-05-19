from sqlalchemy import create_engine, Column, BigInteger, String, Text, ForeignKey, TIMESTAMP, BigInteger as BIGINT
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.schema import Sequence
from datetime import datetime
import os

Base = declarative_base()

class BitrixChat(Base):
    __tablename__ = 'bitrix_chat'
    
    chat_id = Column(BigInteger, primary_key=True, autoincrement=True)
    chat_name = Column(String(255), nullable=False)
    
    # Связи
    chat_users = relationship("ChatUser", back_populates="chat", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<BitrixChat(chat_id={self.chat_id}, chat_name='{self.chat_name}')>"


class BitrixUser(Base):
    __tablename__ = 'bitrix_user'
    
    user_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_name = Column(String(255), nullable=False)
    onec_login = Column(String(50), nullable=False)
    onec_password = Column(String(255), nullable=False)
    
    # Связи
    chat_users = relationship("ChatUser", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<BitrixUser(user_id={self.user_id}, user_name='{self.user_name}')>"


class ChatUser(Base):
    __tablename__ = 'chat_user'
    
    user_id = Column(BigInteger, ForeignKey('bitrix_user.user_id', ondelete='CASCADE'), primary_key=True)
    chat_id = Column(BigInteger, ForeignKey('bitrix_chat.chat_id', ondelete='CASCADE'), primary_key=True)
    message_cnt = Column(BigInteger, nullable=False, default=0)
    token_volume = Column(BigInteger, nullable=False, default=0)
    
    # Связи
    user = relationship("BitrixUser", back_populates="chat_users")
    chat = relationship("BitrixChat", back_populates="chat_users")
    
    def __repr__(self):
        return f"<ChatUser(user_id={self.user_id}, chat_id={self.chat_id})>"


class Request(Base):
    __tablename__ = 'request'
    
    request_id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_text = Column(Text, nullable=False)
    
    # Связи
    messages = relationship("Message", back_populates="request")
    
    def __repr__(self):
        return f"<Request(request_id={self.request_id})>"


class Message(Base):
    __tablename__ = 'message'
    
    message_id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('bitrix_user.user_id', ondelete='RESTRICT'), nullable=False)
    chat_id = Column(BigInteger, ForeignKey('bitrix_chat.chat_id', ondelete='RESTRICT'), nullable=False)
    request_id = Column(BigInteger, ForeignKey('request.request_id', ondelete='RESTRICT'), nullable=True)
    message_text = Column(Text, nullable=False)
    message_date = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    
    # Связи
    user = relationship("BitrixUser", back_populates="messages")
    chat = relationship("BitrixChat", back_populates="messages")
    request = relationship("Request", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(message_id={self.message_id}, user_id={self.user_id})>"