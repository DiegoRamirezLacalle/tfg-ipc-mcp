from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

_client: AsyncIOMotorClient | None = None


def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGO_URI)
    return _client


def get_mongo_db() -> AsyncIOMotorDatabase:
    return _get_client()[settings.MONGO_DB]


async def close_mongo() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
