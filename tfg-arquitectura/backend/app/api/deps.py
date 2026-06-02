from collections.abc import AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.mongo import get_mongo_db
from app.db.postgres import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_mongo() -> AsyncIOMotorDatabase:
    return get_mongo_db()
