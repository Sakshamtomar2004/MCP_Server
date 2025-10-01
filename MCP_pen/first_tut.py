from mcp.server.fastmcp import FastMCP
mcp=FastMCP(name="hello-mcp",stateless_http=True)


@mcp.tool()
def search_online(query:str)->str:
    return f"Searching online for: {query}"

@mcp.tool()
async def get_weather(city:str)->str:
    return f"Fetching weather for: {city}"


mcp_app=mcp.streamable_http_app()

