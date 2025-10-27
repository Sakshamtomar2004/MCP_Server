from fastmcp import FastMCP

mcp=FastMCP.as_proxy(
    "https://mighty-aqua-llama.fastmcp.app/mcp",

    name="Saksham_Tomar_Server"
)

# Add a simple test tool
@mcp.tool()
def say_hello(name: str) -> str:
    """Returns a greeting message."""
    return f"Hello, {name}! 👋 from Saksham_Tomar_Server"


if __name__== "__main__":
    mcp.run()
