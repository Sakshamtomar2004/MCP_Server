from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio
from contextlib import AsyncExitStack

class MCPClient:
    def __init__(self,url):
        self.session=ClientSession(url)
    async def list_tools(self):
            async with self.session as session:
                 response=(await session.list_tools()).tools
                 return response

async def main():
     async with  MCPClient("http://localhost:8000/mcp") as client:
          tools=await client.list_tools()
          for tool in tools:
               print(tool)          

asyncio.run(main())               