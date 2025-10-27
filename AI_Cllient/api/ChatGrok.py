"""
Complete MCP Client with LangGraph Workflow + Grok model
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
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import stdio_client
from langchain_core.prompts import PromptTemplate

# LangGraph
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

# LangChain Grok
from langchain_groq import ChatGroq  # Use Grok LLM

# MCP tools adapter for LangChain
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
    def __init__(self, llm_provider: str = "groq"):
        self.llm_provider = llm_provider
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = None
        self.model_name = "openai/gpt-oss-120b"  # Grok model name
        self.gemini_tools = []
        self.workflow = None
        self.memory = InMemorySaver()
        self.multi_client:Optional[MultiServerMCPClient]=None
        load_dotenv()

    async def connect_to_servers(self, servers: dict) -> bool:
        try:
           # if not os.path.exists(server_script_path):
            #    raise FileNotFoundError(f"MCP script not found: {server_script_path}")
            mcp_configs={}
            for name,path_or_url in servers.items():
                if path_or_url.startswith("https"):
                    mcp_configs[name]={
                        "url":path_or_url,  
                        "transport":"streamable_http"

                    }
                else:
                    if not os.path.exists(path_or_url):
                        raise FileNotFoundError(f"MCP script not found:{path_or_url}")  

                    if os.name == "nt":
                        mcp_configs[name] = {
                            "command": "cmd",
                            "args": ["/c", "python", path_or_url],
                            "transport": "stdio",
                        }
                    else:
                        mcp_configs[name] = {
                            "command": "python",
                            "args": [path_or_url],
                            "transport": "stdio",
                        }  
            self.multi_client=MultiServerMCPClient(mcp_configs)
            self.gemini_tools=await self.multi_client.get_tools()
            logger.info(f"[OK] Loaded {len(self.gemini_tools)} MCP tools from multiple servers.")

                    
            

          #  stdio_transport = await self.exit_stack.enter_async_context(
           #     stdio_client(server_params)
           # )
           # self.stdio, self.write = stdio_transport
            #self.session = await self.exit_stack.enter_async_context(
            #    ClientSession(self.stdio, self.write)
            #)
           # await self.session.initialize()
           # logger.info("[OK] Connected to MCP server")

            # Load MCP tools
           # self.gemini_tools = await load_mcp_tools(self.session)
           # logger.info(f"[OK] Loaded {len(self.gemini_tools)} MCP tools")

            # Initialize Grok LLM
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GROQ_API_KEY in .env")

            self.llm = ChatGroq(
                model=self.model_name,
                groq_api_key=api_key,
                temperature=0.7,
            ).bind_tools(self.gemini_tools)
            logger.info("[OK] Groq LLM configured")

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
            template1 = PromptTemplate(
                template="""You are a professional AI assistant with access to tools for expenses tracking. 
                Maintain a polite and professional tone at all times, regardless of user input.
                If the user uses inappropriate language or is rude, respond politely but firmly that you only engage in professional communication.
                For valid queries, answer the following question concisely:\n{input}""",
                input_variables=['input']
            )
            prompt1 = template1.format_prompt(input=state["input"])
            response = self.llm.invoke(prompt1)
            new_messages = response if isinstance(response, list) else [response]
            return {"messages": new_messages, "llm_response": response}

        
        async def tool_node(state: GraphState):
            logger.info("🧰 Tool Node running...")
            llm_response = state.get("llm_response")
            if not llm_response:
                return {"tools_result": "No LLM response to process."}

            if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
                tool_call = llm_response.tool_calls[0]
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})

                logger.info(f"Calling tool '{tool_name}' with args: {tool_args}")

                try:
                    # Get list of available tools
                    tool = next((t for t in self.gemini_tools if t.name == tool_name), None)
                    if tool:
                        result = await tool.ainvoke(tool_args)
                        result_text = self._extract_result_text(result)
                        return {
                            "tools_result": result_text,
                            "messages": [{"role": "assistant", "content": f"Tool {tool_name} executed: {result_text}"}]
                        }

                    # If tool not found
                    raise ValueError(f"Tool '{tool_name}' not found in available tools.")

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
            def clean(obj):
                if isinstance(obj, list):
                    return [clean(x) for x in obj]
                if hasattr(obj, "content"):
                    # Handle AIMessage and similar objects
                    return {
                        "content": obj.content,
                        "type": obj.__class__.__name__
                    }
                if hasattr(obj, "dict"):
                    try:
                        return obj.dict()
                    except:
                        pass
                if hasattr(obj, "__dict__"):
                    return {k: clean(v) for k, v in obj.__dict__.items() 
                           if not k.startswith('_')}
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                return str(obj)

            result_clean = clean(result_state)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result_clean, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] Conversation saved to {path}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")


    async def cleanup(self):
        try:
            if self.multi_client:
                await self.multi_client.__aexit__(None, None, None)
            await self.exit_stack.aclose()    
            logger.info("[OK] Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def main():
    servers = {
        "campusx": "D:/CA_content/Python/MCP_Server/CampusX/test.py",
        "github": "D:/CA_content/Python/MCP_Server/CampusX/github_mcp.py"
    
    }
    client = MCPWorkflowClient()
    connected = await client.connect_to_servers(servers)
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
            print("\n🤖 Grok Workflow Response:")
            print(response)
            print()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import traceback
    asyncio.run(main())
