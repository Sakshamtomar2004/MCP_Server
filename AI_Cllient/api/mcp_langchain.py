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
from typing import Optional, TypedDict, Annotated
from contextlib import AsyncExitStack

# UTF‑8 console support
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langgraph.schema import add_messages
# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools


# =====================================================
# LOGGING CONFIG
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger("MCPWorkflowClient")


# =====================================================
# DEFINE WORKFLOW STATE
# =====================================================
class GraphState(TypedDict):
    input: str
    output: str
    tools_result: str
    messages: Annotated[list, add_messages]


# =====================================================
# MCP CLIENT WITH LANGGRAPH WORKFLOW
# =====================================================
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

    # -------------------------------------------------
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

            # Initialize Gemini
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

    # -------------------------------------------------
    def _build_workflow(self):
        """Create LangGraph workflow with tool + LLM nodes"""
        graph = StateGraph(GraphState)

        # 1️⃣ LLM Node - Interpret query, decide tool
        def llm_node(state: GraphState):
            logger.info("🤖 LLM Node running...")
            response = self.llm.invoke(state["input"])
            new_messages = response if isinstance(response, list) else [response]
            return {"messages": new_messages}

        # 2️⃣ Tool Node - Execute MCP tools if called
        async def tool_node(state: GraphState):
            logger.info("🧰 Tool Node running...")
            try:
                if not self.gemini_tools:
                    return {"tools_result": "No tools available."}
                # Example: call the first tool as proof-of-concept
                first_tool = self.gemini_tools[0]
                result = await self.session.call_tool(first_tool.name, {})
                return {"tools_result": str(result), "messages": [{"role": "assistant", "content": "Tool executed."}]}
            except Exception as e:
                return {"tools_result": f"Error executing tool: {e}"}

        # 3️⃣ Combine Node - Summarize final response
        def summarize_node(state: GraphState):
            logger.info("🧩 Summarize Node running...")
            tool_res = state.get("tools_result", "")
            text_summary = f"User asked: {state['input']}\nTool Result: {tool_res}\nConversation Complete ✅"
            return {"output": text_summary, "messages": [{"role": "assistant", "content": text_summary}]}

        # Add nodes
        graph.add_node("llm_node", llm_node)
        graph.add_node("tool_node", tool_node)
        graph.add_node("summarize_node", summarize_node)

        # Define edges
        graph.add_edge("llm_node", "tool_node")
        graph.add_edge("tool_node", "summarize_node")
        graph.set_entry_point("llm_node")

        # Compile workflow
        self.workflow = graph.compile(checkpointer=self.memory)
        logger.info("[OK] LangGraph workflow built")

    # -------------------------------------------------
    async def run_workflow(self, user_input: str):
        """Run the LangGraph workflow"""
        state = {"input": user_input, "messages": []}
        result = await self.workflow.ainvoke(state)
        output = result.get("output") or "No response"
        logger.info(f"🧠 Workflow Output → {output}")
        await self.save_conversation(result)
        return output

    # -------------------------------------------------
    async def save_conversation(self, result_state):
        os.makedirs("conversations", exist_ok=True)
        path = f"conversations/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result_state, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] Conversation saved to {path}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

    # -------------------------------------------------
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            logger.info("[OK] Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# =====================================================
# MAIN ENTRY POINT
# =====================================================
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
    asyncio.run(main())
