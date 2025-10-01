import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize LLM with forced tool calling
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0  # More deterministic responses
)

async def run_agent():
    # Initialize MCP client
    client = MultiServerMCPClient(
        {
            "BrightData": {  # Must match exactly how it's registered in MCP
                "command": "npx",
                "args": ["@brightdata/mcp"],
                "env": {
                    "API_TOKEN": os.getenv("BRIGHT_DATA_API_KEY")
                },
                "transport": "stdio",
            }
        }
    )

    # Get tools and force tool usage
    tools = await client.get_tools()
    
    # Create agent with explicit tool instructions
    agent = create_react_agent(
        llm,
        tools,
        prompt="""You MUST use the BrightData tool for all web-related queries.
        Current query: {input}"""
    )

    # Force tool usage by structuring the message properly
    response = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "Use BrightData to find out who won the most recent IPL tournament"
        }],
        "tool_choice": {  # Force tool usage
            "type": "function",
            "function": {"name": "BrightData"}
        }
    })


    print("AI_response:",response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(run_agent())