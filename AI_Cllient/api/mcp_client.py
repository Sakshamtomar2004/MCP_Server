from typing import Optional
from contextlib import AsyncExitStack
import traceback
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from utils.logger import logger
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from anthropic import Anthropic
from anthropic.types import Message
# import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool, content_types
from google import genai
from google.genai import types



class MCPClient:
    def __init__(self, llm_provider: str = "gemini", max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize MCP Client
        
        Args:
            llm_provider: "anthropic" or "gemini"
            max_retries: Number of connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.llm_provider = llm_provider
        self.tools = []
        self.messages = []
        self.logger = logger
        
        # Initialize LLM based on provider
        if llm_provider == "anthropic":
            self.llm = Anthropic()
            self.model_name = "claude-3-5-haiku-20241022"
        elif llm_provider == "gemini":
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model_name = "gemini-flash-latest"  # or use "gemini-2.5-flash"

            self.llm = self.client.models

            self.chat = None
            self.gemini_tools = []
        else:
            raise ValueError("llm_provider must be 'anthropic' or 'gemini'")

    async def connect_to_server(self, server_script_path: str) -> bool:
        """
        Connect to MCP server with retry logic
        
        Args:
            server_script_path: Path to the MCP server script
            
        Returns:
            bool: True if connection successful
        """
        # Validate path exists
        if not os.path.exists(server_script_path):
            self.logger.error(f"Server script path does not exist: {server_script_path}")
            raise FileNotFoundError(f"MCP server script not found: {server_script_path}")

        # Parse file and command
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        
        # Retry connection logic
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"Connecting to MCP server (attempt {attempt}/{self.max_retries})")
                self.logger.info(f"Server script: {server_script_path}")
                
                server_params = StdioServerParameters(
                    command=command,
                    args=[server_script_path],
                    env=None
                )

                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                self.stdio, self.write = stdio_transport
                
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.write)
                )

                # Initialize with timeout
                try:
                    await asyncio.wait_for(self.session.initialize(), timeout=10.0)
                    self.logger.info("✓ Connected to MCP server successfully")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Connection timeout on attempt {attempt}")
                    await self.exit_stack.aclose()
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(self.retry_delay)
                    continue

                # Get tools
                mcp_tools = await self.get_mcp_tools()
                self.tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in mcp_tools
                ]

                self.logger.info(f"Available tools: {[tool['name'] for tool in self.tools]}")

                # Convert tools based on provider
                if self.llm_provider == "gemini":
                    self.logger.info("Converting tools to Gemini format...")
                    self.gemini_tools = self._convert_to_gemini_tools(self.tools)
                   # self.chat = self.llm.start_chat(enable_automatic_function_calling=False)
                    self.chat = self.client.chats.create(model=self.model_name)

                    self.logger.info(f"✓ Converted {len(self.gemini_tools[0].function_declarations)} tools for Gemini")

                return True

            except Exception as e:
                self.logger.error(f"Connection attempt {attempt} failed: {e}")
                traceback.print_exc()
                
                # Try to cleanup
                try:
                    await self.exit_stack.aclose()
                except:
                    pass
                
                # If last attempt, raise error
                if attempt == self.max_retries:
                    self.logger.error(f"Failed to connect after {self.max_retries} attempts")
                    raise
                
                # Wait before retrying
                self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)

    def _convert_to_gemini_tools(self, mcp_tools):
        """
        Convert MCP tools to Gemini format
        
        Gemini's Schema only supports: type, format, description, enum, items, properties, required
        It does NOT support: title, additionalProperties, examples, default, etc.
        """
        function_declarations = []
        
        for tool in mcp_tools:
            # Get the input schema and sanitize it for Gemini
            input_schema = tool.get("input_schema", {})
            gemini_schema = self._sanitize_schema_for_gemini(input_schema)
            
            try:
                function_declaration = FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=gemini_schema
                )
                function_declarations.append(function_declaration)
                self.logger.info(f"✓ Converted tool: {tool['name']}")
            except Exception as e:
                self.logger.warning(f"⚠ Failed to convert tool {tool['name']}: {e}")
                # Continue with other tools even if one fails
                continue
        
        return [Tool(function_declarations=function_declarations)] if function_declarations else []

    def _sanitize_schema_for_gemini(self, schema):
        """
        Sanitize JSON Schema to be compatible with Gemini's Schema format
        
        Removes unsupported fields and ensures proper structure
        """
        if not schema:
            return {}
        
        # Create a copy to avoid modifying the original
        sanitized = {}
        
        # Supported fields in Gemini Schema
        supported_fields = {'type', 'format', 'description', 'enum', 'items', 'properties', 'required'}

        # Copy only supported top-level fields
        for field in supported_fields:
            if field in schema:
                if field == 'properties':
                    # Recursively sanitize nested properties
                    sanitized['properties'] = {
                        key: self._sanitize_schema_for_gemini(value)
                        for key, value in schema['properties'].items()
                    }
                elif field == 'items':
                    # Recursively sanitize array items
                    sanitized['items'] = self._sanitize_schema_for_gemini(schema['items'])
                else:
                    sanitized[field] = schema[field]
        
        # If schema has properties but no type specified, set to 'object'
        if 'properties' in sanitized and 'type' not in sanitized:
            sanitized['type'] = 'object'
        
        return sanitized

    async def get_mcp_tools(self):
        """Get list of available MCP tools"""
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    async def process_query(self, query: str):
        """Process query using configured LLM"""
        try:
            self.logger.info(f"Processing query: {query}")
            
            if self.llm_provider == "anthropic":
                return await self._process_query_anthropic(query)
            else:
                return await self._process_query_gemini(query)

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    async def _process_query_anthropic(self, query: str):
        """Process query using Anthropic"""
        user_message = {"role": "user", "content": query}
        self.messages = [user_message]

        while True:
            response = await self.call_llm()

            if response.content[0].type == "text" and len(response.content) == 1:
                assistant_message = {
                    "role": "assistant",
                    "content": response.content[0].text,
                }
                self.messages.append(assistant_message)
                await self.log_conversation()
                break

            assistant_message = {
                "role": "assistant",
                "content": response.to_dict()["content"],
            }
            self.messages.append(assistant_message)
            await self.log_conversation()

            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input
                    tool_use_id = content.id
                    self.logger.info(f"Calling tool {tool_name} with args {tool_args}")
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        self.logger.info(f"Tool {tool_name} result: {result}...")
                        self.messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": result.content,
                                    }
                                ],
                            }
                        )
                        await self.log_conversation()
                    except Exception as e:
                        self.logger.error(f"Error calling tool {tool_name}: {e}")
                        raise

        return self.messages

    async def _process_query_gemini(self, query: str):
        """Process query using Gemini"""
        conversation_history = [{"role": "user", "content": query}]
        current_query = query

        while True:
            response = await self.call_llm(current_query)

            if response.candidates and response.candidates[0].content.parts:
                has_function_call = any(
                    hasattr(part, 'function_call') and part.function_call
                    for part in response.candidates[0].content.parts
                )

                if not has_function_call:
                    final_text = response.text
                    conversation_history.append({
                        "role": "model",
                        "content": final_text
                    })
                    self.logger.info(f"Final response: {final_text}")
                    self.messages = conversation_history
                    await self.log_conversation()
                    break

                tool_results = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        tool_name = function_call.name
                        tool_args = dict(function_call.args)

                        self.logger.info(f"Calling tool {tool_name} with args {tool_args}")

                        try:
                            result = await self.session.call_tool(tool_name, tool_args)
                            result_content = self._extract_tool_result(result)
                            self.logger.info(f"Tool {tool_name} result: {str(result_content)[:100]}")

                            tool_results.append({
                                "function_call": function_call,
                                "result": result_content
                            })

                        except Exception as e:
                            self.logger.error(f"Error calling tool {tool_name}: {e}")
                            tool_results.append({
                                "function_call": function_call,
                                "result": f"Error: {str(e)}"
                            })

                if tool_results:
                    response_parts = []
                    for tr in tool_results:
                        response_parts.append(
                            types.Part.from_function_response(
                                name=tr["function_call"].name,
                                response={"result": tr["result"]}
                            )
                        )
                    
                    current_query = response_parts
                else:
                    break
            else:
                break

        self.messages = conversation_history
        return conversation_history

    def _extract_tool_result(self, result):
        """Extract content from MCP tool result"""
        if hasattr(result, 'content'):
            if isinstance(result.content, list):
                return [
                    item.text if hasattr(item, 'text') else str(item)
                    for item in result.content
                ]
            return str(result.content)
        return str(result)

    async def call_llm(self, message=None):
        """Call LLM with proper error handling"""
        try:
            self.logger.info(f"Calling LLM ({self.llm_provider})")
            
            if self.llm_provider == "anthropic":
                return self.llm.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    messages=self.messages,
                    tools=self.tools,
                )
            else:  # gemini
                if isinstance(message, list):
    # list of parts (tool responses, etc.)
                    content_input = message
                else:
    # simple text query
                    content_input = [{"role": "user", "parts": [{"text": message}]}]

# Use the modern Gemini generate_content API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=content_input,
                    config=types.GenerateContentConfig(
                    tools=self.gemini_tools
                 ) if self.gemini_tools else None,
)

                return response














                
                
                
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.exit_stack.aclose()
            self.logger.info("✓ Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()

    async def log_conversation(self):
        """Log conversation to JSON file"""
        os.makedirs("conversations", exist_ok=True)

        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message["role"], "content": []}

                if isinstance(message["content"], str):
                    serializable_message["content"] = message["content"]
                elif isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["content"].append(
                                content_item.to_dict()
                            )
                        elif hasattr(content_item, "dict"):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["content"].append(
                                content_item.model_dump()
                            )
                        else:
                            serializable_message["content"].append(content_item)

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
