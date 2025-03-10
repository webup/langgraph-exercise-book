# [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

This library provides a lightweight wrapper that makes [Anthropic Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) tools compatible with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph).

## Features

- 🛠️ Convert MCP tools into [LangChain tools](https://python.langchain.com/docs/concepts/tools/) that can be used with [LangGraph](https://github.com/langchain-ai/langgraph) agents
- 📦 A client implementation that allows you to connect to multiple MCP servers and load tools from them

## Quickstart

Here is a simple example of using the MCP tools with a LangGraph agent.

```bash
pip install langchain-mcp-adapters langgraph langchain-openai

export OPENAI_API_KEY=<your_api_key>
export OPENAI_API_BASE=<your_api_base>
```

### Server

First, let's create an MCP server that can add and multiply numbers.

```py
# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Client

👇 LangSmith Trace: https://smith.langchain.com/public/31317cdf-2716-4561-8532-370fdd5add35/r

> You might notice the tool calling sequence is not correct; this is due to the model's limitations.


```python
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct")

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/Users/haili/workspaces/langgraph-exercise-book/langchain-mcp-adapters/math_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools
        tools = await load_mcp_tools(session)

        # Create and run the agent
        agent = create_react_agent(model, tools)
        agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        print(agent_response["messages"][-1].content)
```

    The result of (3 + 5) x 12 is 96.


## Multiple MCP Servers

The library also allows you to connect to multiple MCP servers and load tools from them:

### Server

```python
# math_server.py
...

# weather_server.py
from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> int:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="sse")
```

```bash
python weather_server.py
```

After running the weather server and invoking the agent, you should see the following output:

```bash
❯ python ./weather_server.py
INFO:     Started server process [73767]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:65140 - "GET /sse HTTP/1.1" 200 OK
INFO:     127.0.0.1:65142 - "POST /messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:65144 - "POST /messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd HTTP/1.1" 202 Accepted
INFO:     127.0.0.1:65146 - "POST /messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd HTTP/1.1" 202 Accepted
Processing request of type ListToolsRequest
INFO:     127.0.0.1:65154 - "POST /messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd HTTP/1.1" 202 Accepted
Processing request of type CallToolRequest
```

👇 LangSmith Trace (Weather Server): https://smith.langchain.com/public/143a5a8b-8f23-4cdc-bf12-c0471bbe0280/r


```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct")

async with MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["/Users/haili/workspaces/langgraph-exercise-book/langchain-mcp-adapters/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # make sure you start your weather server on port 8000
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        }
    }
) as client:
    agent = create_react_agent(model, client.get_tools())

    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    print(math_response["messages"][-1].content)

    weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
    print(weather_response["messages"][-1].content)
```

    Connecting to SSE endpoint: http://localhost:8000/sse
    HTTP Request: GET http://localhost:8000/sse "HTTP/1.1 200 OK"
    Received endpoint URL: http://localhost:8000/messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd
    Starting post writer with endpoint URL: http://localhost:8000/messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd
    HTTP Request: POST http://localhost:8000/messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd "HTTP/1.1 202 Accepted"
    HTTP Request: POST http://localhost:8000/messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd "HTTP/1.1 202 Accepted"
    HTTP Request: POST http://localhost:8000/messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd "HTTP/1.1 202 Accepted"
    HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
    HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"


    The result of (3 + 5) x 12 is 96.


    HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"
    HTTP Request: POST http://localhost:8000/messages/?session_id=62ea9f2834d944caa1bfc888a30ffcdd "HTTP/1.1 202 Accepted"
    HTTP Request: POST https://api.siliconflow.cn/v1/chat/completions "HTTP/1.1 200 OK"


    It seems there was an issue retrieving the exact weather data for NYC. However, the message received suggests it's currently sunny in New York. For the most accurate and up-to-date weather information, please check a reliable weather website or app.

