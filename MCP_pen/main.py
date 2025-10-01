import asyncio


async def get_connections(name):
    class ctx():
        async def __aenter__(self):
            print(f"connecting...{name}")
            return name
        async def __aexit__(self,exc_type,exc,tb):
            print(f"Connected!{name}")
    return ctx()    


#asynchronous function ka return await lagakar hi lena hota h
async def main():
    async with await get_connections("Saksham") as a:
        async with await get_connections("Shivendra") as b:
           print(f"Using connection:{a} and{b}")

asyncio.run(main())        