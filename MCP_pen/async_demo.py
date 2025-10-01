import asyncio
from contextlib import asynccontextmanager 


@asynccontextmanager
async def make_connection(name):
    print(f"Connecting...{name}")
    yield name
    print(f"Connected{name}")


async def main():
    async with make_connection("Saksham") as conn:
        print(f"Using connection {conn}")

asyncio.run(main())