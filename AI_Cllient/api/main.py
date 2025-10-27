from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from mcp_client import MCPClient
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Configuration settings"""
    server_script_path: str = "D:/CA_content/Python/MCP_Server/CampusX/test.py"
    default_llm_provider: str = "gemini"  # "gemini" or "anthropic"
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    model_config = ConfigDict(
        env_file=".env",
        extra="ignore"  # Ignore any extra fields not declared
    )


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    client = None
    try:
        # Initialize MCPClient with default LLM provider
        client = MCPClient(llm_provider=settings.default_llm_provider)
        
        # Connect to MCP server
        connected = await client.connect_to_server(settings.server_script_path)
        if not connected:
            raise HTTPException(
                status_code=500, 
                detail="Failed to connect to MCP server"
            )
        
        app.state.client = client
        app.state.current_llm_provider = settings.default_llm_provider
        print(f"✓ Connected to MCP server with {settings.default_llm_provider} LLM")
        
        yield
        
    except Exception as e:
        print(f"Error during lifespan startup: {e}")
        raise HTTPException(status_code=500, detail="Error during lifespan") from e
    
    finally:
        # Cleanup
        if client:
            await client.cleanup()
            print("✓ Disconnected from MCP server")


app = FastAPI(
    title="MCP Client API",
    description="FastAPI wrapper for MCP Client with Gemini and Anthropic support",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============

class QueryRequest(BaseModel):
    """Query request model"""
    query: str
    llm_provider: Optional[str] = None  # Override default provider


class Message(BaseModel):
    """Message model"""
    role: str
    content: Any


class ToolCall(BaseModel):
    """Tool call model"""
    name: str
    args: Dict[str, Any]


class ToolInfo(BaseModel):
    """Tool information model"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class QueryResponse(BaseModel):
    """Query response model"""
    messages: list
    llm_provider: str


class ToolsResponse(BaseModel):
    """Tools response model"""
    tools: list


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "MCP Client API",
        "version": "1.0.0",
        "llm_provider": app.state.current_llm_provider,
        "endpoints": {
            "query": "/query (POST) - Process a query",
            "tools": "/tools (GET) - Get available tools",
            "switch-provider": "/switch-provider (POST) - Switch LLM provider",
            "info": "/info (GET) - Get current configuration"
        }
    }


@app.post("/query")
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Process a query and return the response
    
    Supports switching LLM provider per request:
    - llm_provider: "gemini" or "anthropic"
    """
    try:
        # Check if provider needs to be switched
        if request.llm_provider and request.llm_provider != app.state.current_llm_provider:
            await switch_llm_provider(request.llm_provider)
        
        # Process the query
        messages = await app.state.client.process_query(request.query)
        
        return QueryResponse(
            messages=messages,
            llm_provider=app.state.current_llm_provider
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def get_tools() -> ToolsResponse:
    """Get the list of available tools from MCP server"""
    try:
        tools = await app.state.client.get_mcp_tools()
        return ToolsResponse(
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]
        )
    except Exception as e:
        print(f"Error getting tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch-provider")
async def switch_provider(provider: str):
    """
    Switch between LLM providers
    
    Parameters:
    - provider: "gemini" or "anthropic"
    """
    try:
        if provider not in ["gemini", "anthropic"]:
            raise ValueError("Provider must be 'gemini' or 'anthropic'")
        
        await switch_llm_provider(provider)
        
        return {
            "message": f"Successfully switched to {provider}",
            "current_provider": app.state.current_llm_provider
        }
        
    except Exception as e:
        print(f"Error switching provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    """Get current configuration and status"""
    try:
        client = app.state.client
        tools = await client.get_mcp_tools()
        
        return {
            "status": "running",
            "current_llm_provider": app.state.current_llm_provider,
            "mcp_server": settings.server_script_path,
            "available_tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                }
                for tool in tools
            ],
            "tool_count": len(tools)
        }
    except Exception as e:
        print(f"Error getting info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provider")
async def get_current_provider():
    """Get current LLM provider"""
    return {
        "current_provider": app.state.current_llm_provider,
        "available_providers": ["gemini", "anthropic"]
    }


# ============== Helper Functions ==============

async def switch_llm_provider(new_provider: str):
    """Switch to a different LLM provider"""
    try:
        if new_provider == app.state.current_llm_provider:
            return  # Already using this provider
        
        # Create new client with different provider
        new_client = MCPClient(llm_provider=new_provider)
        
        # Connect to MCP server
        connected = await new_client.connect_to_server(settings.server_script_path)
        if not connected:
            raise Exception("Failed to connect to MCP server")
        
        # Cleanup old client
        old_client = app.state.client
        await old_client.cleanup()
        
        # Update app state
        app.state.client = new_client
        app.state.current_llm_provider = new_provider
        
        print(f"✓ Switched to {new_provider} LLM provider")
        
    except Exception as e:
        print(f"Error switching provider: {e}")
        raise


# ============== Error Handlers ==============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info"
    )
