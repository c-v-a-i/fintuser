from prisma import Prisma


async def get_prisma_db() -> Prisma:
    """Connect to the Prisma-managed database."""
    db = Prisma()
    await db.connect()
    return db


async def disconnect_db(db: Prisma) -> None:
    """Disconnect from the database."""
    await db.disconnect()
