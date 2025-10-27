"""
Complete MCP Client with LangGraph Workflow + Gemini 2.5 Flash
Fully compatible with MCP SDK (>=0.3.8) and Windows
"""

import os
import sys
import asyncio
import json
import traceback
import logging
from datetime import datetime
from typing import Optional, TypedDict
from contextlib import AsyncExitStack
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# LangGraph
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger("MCPWorkflowClient")

# Define Workflow State
class GraphState(TypedDict, total=False):
    input: str
    output: str
    tools_result: str
    messages: list
    llm_response: Optional[object]  # store LLM response for tool call extraction

class MCPWorkflowClient:
    def __init__(self, llm_provider: str = "gemini"):
        self.llm_provider = llm_provider
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = None
        self.model_name = "gemini-2.5-flash"
        self.gemini_tools = []
        self.workflow = None
        self.memory = InMemorySaver()
        load_dotenv()

    async def connect_to_server(self, server_script_path: str) -> bool:
        try:
            if not os.path.exists(server_script_path):
                raise FileNotFoundError(f"MCP script not found: {server_script_path}")

            if os.name == "nt":
                server_params = StdioServerParameters(
                    command="cmd",
                    args=["/c", "python", server_script_path],
                )
            else:
                server_params = StdioServerParameters(
                    command="python",
                    args=[server_script_path],
                )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()
            logger.info("[OK] Connected to MCP server")

            # Load MCP tools
            self.gemini_tools = await load_mcp_tools(self.session)
            logger.info(f"[OK] Loaded {len(self.gemini_tools)} MCP tools")

            # Initialize Gemini LLM
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GOOGLE_API_KEY in .env")

            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.7,
            ).bind_tools(self.gemini_tools)
            logger.info("[OK] Gemini LLM configured")

            # Build LangGraph workflow
            self._build_workflow()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            traceback.print_exc()
            return False

    def _build_workflow(self):
        graph = StateGraph(GraphState)

        def llm_node(state: GraphState):
            logger.info("🤖 LLM Node running...")
            response = self.llm.invoke(state["input"])
            new_messages = response if isinstance(response, list) else [response]
            return {"messages": new_messages, "llm_response": response}

        async def tool_node(state: GraphState):
            logger.info("🧰 Tool Node running...")
            llm_response = state.get("llm_response")
            if not llm_response:
                return {"tools_result": "No LLM response to process."}
            
            # Check if Gemini LLM made a tool call
            if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
                tool_call = llm_response.tool_calls[0]  # Take first tool call
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                logger.info(f"Calling tool '{tool_name}' with args: {tool_args}")

                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    result_text = self._extract_result_text(result)
                    return {
                        "tools_result": result_text,
                        "messages": [{"role": "assistant", "content": f"Tool {tool_name} executed: {result_text}"}]
                    }
                except Exception as e:
                    logger.error(f"Error calling tool '{tool_name}': {e}")
                    return {"tools_result": f"Error executing {tool_name}: {e}"}
            else:
                return {"tools_result": llm_response.content if hasattr(llm_response, "content") else str(llm_response)}

        def summarize_node(state: GraphState):
            logger.info("🧩 Summarize Node running...")
            tool_res = state.get("tools_result", "")
            text_summary = f"{tool_res}\nConversation Complete ✅"
            return {"output": text_summary, "messages": [{"role": "assistant", "content": text_summary}]}

        graph.add_node("llm_node", llm_node)
        graph.add_node("tool_node", tool_node)
        graph.add_node("summarize_node", summarize_node)

        graph.add_edge("llm_node", "tool_node")
        graph.add_edge("tool_node", "summarize_node")
        graph.set_entry_point("llm_node")

        self.workflow = graph.compile(checkpointer=self.memory)
        logger.info("[OK] LangGraph workflow built")

    def _extract_result_text(self, result):
        if hasattr(result, "content"):
            if isinstance(result.content, list):
                return "\n".join(
                    [item.text if hasattr(item, "text") else str(item) for item in result.content]
                )
            return str(result.content)
        return str(result)

    async def run_workflow(self, user_input: str):
        state = {"input": user_input, "messages": []}
        config = {"configurable": {"thread_id": "main_thread"}}
        result = await self.workflow.ainvoke(state, config)
        output = result.get("output") or "No response"
        logger.info(f"🧠 Workflow Output → {output}")
        await self.save_conversation(result)
        return output

    async def save_conversation(self, result_state):
        os.makedirs("conversations", exist_ok=True)
        path = f"conversations/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result_state, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] Conversation saved to {path}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            logger.info("[OK] Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    server = "D:/CA_content/Python/MCP_Server/CampusX/test.py"
    client = MCPWorkflowClient()
    connected = await client.connect_to_server(server)
    if not connected:
        print("❌ Connection failed.")
        return

    print("✅ Connected to MCP server. Type queries or 'exit'.")

    try:
        while True:
            user_inp = input(">>> ").strip()
            if not user_inp or user_inp.lower() in ("exit", "quit"):
                break
            response = await client.run_workflow(user_inp)
            print("\n🤖 Gemini Workflow Response:")
            print(response)
            print()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import traceback
    asyncio.run(main())
