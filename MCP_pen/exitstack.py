import asyncio
from contextlib import AsyncExitStack

async def get_connections(name):
    class ctx():
        async def __aenter__(self):
            print(f"connecting...{name}")
            return name
        async def __aexit__(self, exc_type, exc, tb):
            print(f"Connected! {name}")  # Fixed typo (added space)
    return ctx()  # Returns the context manager (no await needed)

async def main():
    async with AsyncExitStack() as stack:
        a = await stack.enter_async_context(await get_connections("A"))  # Removed inner await
        b = await stack.enter_async_context(await get_connections("B"))  # Removed inner await
        print(f"Using connections: {a} and {b}")  # Fixed typo (added space)
        
        def customCleanup():
            print("Custom clean up logic")
        stack.push_async_callback(customCleanup)
        print(f"Doing work with{a}")    


asyncio.run(main())

#it make sure ki connections saare close bhi ho gye h

