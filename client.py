import asyncio
from mcp.client.stdio import StdioClient  # Correct import for stdio transport

async def main():
    # Create connection to the math server
    math_connection = StdioClient(
        command="python",
        args=["add.py"]
    )
    
    try:
        # Initialize the connection
        await math_connection.connect()
        
        # Get available tools
        tools = await math_connection.get_tools()
        print("Available tools:", list(tools.keys()))
        
        # Test the add function
        result = await tools["add"].invoke({"a": 3, "b": 5})
        print("3 + 5 =", result)
        
    except Exception as e:
        print("Error:", e)
    finally:
        # Close the connection
        await math_connection.close()

asyncio.run(main())