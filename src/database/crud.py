"""
CRUD-операции для работы с базой данных чат-бота.
Все функции управляют собственной сессией (SessionLocal) для атомарности.
"""
import logging
from typing import Optional
from datetime import datetime

from src.database.database import SessionLocal
from src.database.models import BitrixChat, BitrixUser, ChatUser, Request, Message

logger = logging.getLogger(__name__)


# ==================== BitrixUser ====================

def get_or_create_user(
    user_id: int, 
    user_name: str, 
    onec_login: str = "", 
    onec_password: str = ""
) -> Optional[BitrixUser]:
    """
    Получает пользователя из БД или создаёт нового, если не найден.
    
    Returns:
        BitrixUser или None при ошибке.
    """
    db = SessionLocal()
    try:
        user = db.query(BitrixUser).filter(BitrixUser.user_id == user_id).first()
        if not user:
            user = BitrixUser(
                user_id=user_id,
                user_name=user_name,
                onec_login=onec_login,
                onec_password=onec_password
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Пользователь {user_name} (ID: {user_id}) добавлен в БД")
        elif user.user_name != user_name:
            user.user_name = user_name
            db.commit()
        return user
    except Exception as e:
        logger.error(f"Ошибка get_or_create_user: {e}")
        db.rollback()
        return None
    finally:
        db.close()


# ==================== BitrixChat ====================

def get_or_create_chat(
    chat_id: str, 
    chat_name: Optional[str] = None
) -> Optional[BitrixChat]:
    """
    Получает чат из БД или создаёт новый, если не найден.
    
    Returns:
        BitrixChat или None при ошибке.
    """
    db = SessionLocal()
    try:
        chat = db.query(BitrixChat).filter(BitrixChat.chat_id == chat_id).first()
        if not chat:
            chat = BitrixChat(
                chat_id=chat_id,
                chat_name=chat_name or f"Chat_{chat_id}"
            )
            db.add(chat)
            db.commit()
            db.refresh(chat)
            logger.info(f"Чат {chat_id} добавлен в БД")
        return chat
    except Exception as e:
        logger.error(f"Ошибка get_or_create_chat: {e}")
        db.rollback()
        return None
    finally:
        db.close()


# ==================== Request (1C-запросы) ====================

def create_1c_query(query_text: str) -> Optional[int]:
    """
    Сохраняет сгенерированный 1C SQL-запрос в таблицу request.
    
    Returns:
        request_id (int) или None при ошибке.
    """
    if not query_text:
        return None
        
    db = SessionLocal()
    try:
        request = Request(request_text=query_text)
        db.add(request)
        db.commit()
        db.refresh(request)
        logger.debug(f"1C-запрос сохранён в БД (request_id={request.request_id})")
        return request.request_id
    except Exception as e:
        logger.error(f"Ошибка create_1c_query: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def get_request_by_id(request_id: int) -> Optional[Request]:
    """Получает запись запроса по ID."""
    db = SessionLocal()
    try:
        return db.query(Request).filter(Request.request_id == request_id).first()
    except Exception as e:
        logger.error(f"Ошибка get_request_by_id: {e}")
        return None
    finally:
        db.close()


# ==================== Message ====================

def create_message(
    user_id: int,
    chat_id: str,
    user_message: str,
    request_id: Optional[int] = None
) -> Optional[Message]:
    """
    Сохраняет сообщение пользователя в таблицу message.
    
    Args:
        user_id: ID пользователя
        chat_id: ID чата
        user_message: Текст сообщения от пользователя
        request_id: ID связанного 1C-запроса (может быть None)
    
    Returns:
        Message или None при ошибке.
    """
    db = SessionLocal()
    try:
        message = Message(
            user_id=user_id,
            chat_id=chat_id,
            request_id=request_id,
            message_text=user_message,
            message_date=datetime.utcnow()
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        logger.debug(f"Сообщение пользователя сохранено в БД (request_id={request_id})")
        return message
    except Exception as e:
        logger.error(f"Ошибка create_message: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def get_messages_by_user(
    user_id: int, 
    limit: int = 50,
    with_requests_only: bool = False
) -> list[Message]:
    """
    Получает сообщения пользователя.
    
    Args:
        user_id: ID пользователя
        limit: Максимальное количество записей
        with_requests_only: Если True, возвращать только сообщения с request_id
    
    Returns:
        Список Message.
    """
    db = SessionLocal()
    try:
        query = db.query(Message).filter(Message.user_id == user_id)
        if with_requests_only:
            query = query.filter(Message.request_id.isnot(None))
        return query.order_by(Message.message_date.desc()).limit(limit).all()
    except Exception as e:
        logger.error(f"Ошибка get_messages_by_user: {e}")
        return []
    finally:
        db.close()


# ==================== ChatUser (статистика) ====================

def update_chat_user_stats(
    user_id: int,
    chat_id: str,
    message_length: int
) -> bool:
    """
    Обновляет или создаёт запись статистики для пары пользователь-чат.
    
    Returns:
        True при успехе, False при ошибке.
    """
    db = SessionLocal()
    try:
        chat_user = db.query(ChatUser).filter(
            ChatUser.user_id == user_id,
            ChatUser.chat_id == chat_id
        ).first()
        
        if chat_user:
            chat_user.message_cnt += 1
            chat_user.token_volume += message_length
        else:
            chat_user = ChatUser(
                user_id=user_id,
                chat_id=chat_id,
                message_cnt=1,
                token_volume=message_length
            )
            db.add(chat_user)
        
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Ошибка update_chat_user_stats: {e}")
        db.rollback()
        return False
    finally:
        db.close()


# ==================== Composite: логирование взаимодействия ====================

def log_user_interaction(
    user_id: int,
    chat_id: str,
    user_name: str,
    user_question: str,
    generated_query: Optional[str] = None,
    existing_request_id: Optional[int] = None,
    success: bool = True,
    error_text: Optional[str] = None,
    onec_login: str = "",
    onec_password: str = ""
) -> bool:
    """
    Композитная операция: сохраняет взаимодействие пользователя с ботом.
    
    Логика:
    1. Гарантирует наличие пользователя и чата в БД.
    2. Если есть generated_query и нет existing_request_id → сохраняет новый запрос.
    3. Сохраняет сообщение пользователя с привязкой к request_id (или NULL).
    4. Обновляет статистику в chat_user.
    
    Returns:
        True при успехе, False при ошибке.
    """
    db = SessionLocal()
    try:
        # 1. Пользователь
        user = db.query(BitrixUser).filter(BitrixUser.user_id == user_id).first()
        if not user:
            user = BitrixUser(
                user_id=user_id,
                user_name=user_name,
                onec_login=onec_login,
                onec_password=onec_password
            )
            db.add(user)
            db.flush()  # Получаем ID без коммита
        elif user.user_name != user_name:
            user.user_name = user_name
        
        # 2. Чат
        chat = db.query(BitrixChat).filter(BitrixChat.chat_id == chat_id).first()
        if not chat:
            chat = BitrixChat(
                chat_id=chat_id,
                chat_name=f"Chat_{chat_id}"
            )
            db.add(chat)
            db.flush()
        
        # 3. 1C-запрос (только если новый)
        final_request_id = existing_request_id
        if final_request_id is None and generated_query and not generated_query.startswith("ОШИБКА:"):
            new_request = Request(request_text=generated_query)
            db.add(new_request)
            db.flush()
            final_request_id = new_request.request_id
        
        # 4. Сообщение пользователя
        message_text = user_question
        if error_text and not success:
            message_text = f"[ERROR] {error_text}\n\n{user_question}"
            
        message = Message(
            user_id=user.user_id,
            chat_id=chat.chat_id,
            request_id=final_request_id,
            message_text=message_text,
            message_date=datetime.utcnow()
        )
        db.add(message)
        
        # 5. Статистика
        chat_user = db.query(ChatUser).filter(
            ChatUser.user_id == user.user_id,
            ChatUser.chat_id == chat.chat_id
        ).first()
        
        if chat_user:
            chat_user.message_cnt += 1
            chat_user.token_volume += len(user_question)
        else:
            db.add(ChatUser(
                user_id=user.user_id,
                chat_id=chat.chat_id,
                message_cnt=1,
                token_volume=len(user_question)
            ))
        
        db.commit()
        logger.debug(f"Взаимодействие залогировано (request_id={final_request_id})")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка log_user_interaction: {e}")
        db.rollback()
        return False
    finally:
        db.close()