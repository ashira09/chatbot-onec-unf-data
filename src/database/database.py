from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base
import os

# Получаем данные для подключения из переменных окружения
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/chatbot_db"
)

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Создает все таблицы в базе данных"""
    print("Создание таблиц...")
    Base.metadata.create_all(bind=engine)
    print("Таблицы успешно созданы!")

def get_db():
    """Получение сессии базы данных"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()