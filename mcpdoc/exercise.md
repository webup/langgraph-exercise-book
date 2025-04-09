# MCP LLMS-TXT Documentation Server

## Overview

[llms.txt](https://llmstxt.org/) is a website index for LLMs, providing background information, guidance, and links to detailed markdown files. IDEs like Cursor and Windsurf or apps like Claude Code/Desktop can use `llms.txt` to retrieve context for tasks. However, these apps use different built-in tools to read and process files like `llms.txt`. The retrieval process can be opaque, and there is not always a way to audit the tool calls or the context returned.

[MCP](https://github.com/modelcontextprotocol) offers a way for developers to have *full control* over tools used by these applications. Here, we create [an open source MCP server](https://github.com/modelcontextprotocol) to provide MCP host applications (e.g., Cursor, Windsurf, Claude Code/Desktop) with (1) a user-defined list of `llms.txt` files and (2) a simple  `fetch_docs` tool read URLs within any of the provided `llms.txt` files. This allows the user to audit each tool call as well as the context returned. 

![](https://github.com/user-attachments/assets/736f8f55-833d-4200-b833-5fca01a09e1b)

## llms-txt

You can find llms.txt files for langgraph and langchain here:

| Library          | llms.txt                                                                                                   |
|------------------|------------------------------------------------------------------------------------------------------------|
| LangGraph Python | [https://langchain-ai.github.io/langgraph/llms.txt](https://langchain-ai.github.io/langgraph/llms.txt)     |
| LangGraph JS     | [https://langchain-ai.github.io/langgraphjs/llms.txt](https://langchain-ai.github.io/langgraphjs/llms.txt) |
| LangChain Python | [https://python.langchain.com/llms.txt](https://python.langchain.com/llms.txt)                             |
| LangChain JS     | [https://js.langchain.com/llms.txt](https://js.langchain.com/llms.txt)                                     |

## Quickstart

### Dry run at local

#### Install uv
Please see [official uv docs](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) for other ways to install `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Choose an `llms.txt` file to use. 
For example, [here's](https://langchain-ai.github.io/langgraph/llms.txt) the LangGraph `llms.txt` file.

> **Note: Security and Domain Access Control**
> 
> For security reasons, mcpdoc implements strict domain access controls:
> 
> 1. **Remote llms.txt files**: When you specify a remote llms.txt URL (e.g., `https://langchain-ai.github.io/langgraph/llms.txt`), mcpdoc automatically adds only that specific domain (`langchain-ai.github.io`) to the allowed domains list. This means the tool can only fetch documentation from URLs on that domain.
> 
> 2. **Local llms.txt files**: When using a local file, NO domains are automatically added to the allowed list. You MUST explicitly specify which domains to allow using the `--allowed-domains` parameter.
> 
> 3. **Adding additional domains**: To allow fetching from domains beyond those automatically included:
>    - Use `--allowed-domains domain1.com domain2.com` to add specific domains
>    - Use `--allowed-domains '*'` to allow all domains (use with caution)
> 
> This security measure prevents unauthorized access to domains not explicitly approved by the user, ensuring that documentation can only be retrieved from trusted sources.]

#### Test the MCP server locally with your `llms.txt` file(s) of choice:

```bash
uvx --from mcpdoc mcpdoc \
     --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" "LangChain:https://python.langchain.com/llms.txt" \
     --transport sse \
     --port 8082 \
     --host localhost
```

> This command starts the MCP documentation server with the following parameters:
> - `--from mcpdoc mcpdoc`: Installs and runs the mcpdoc package
> - `--urls`: Specifies the llms.txt files to use, with labels "LangGraph" and "LangChain"
> - `--transport sse`: Uses Server-Sent Events for communication
> - `--port 8082`: Runs the server on port 8082
> - `--host localhost`: Makes the server available on localhost
>
> The MCP server acts as a bridge between LLM applications and documentation sources, giving you full visibility and control over what context is being retrieved.

❯ uvx --from mcpdoc mcpdoc \
     --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" "LangChain:https://python.langchain.com/llms.txt" \
     --transport sse \
     --port 8082 \
     --host localhost

    ███╗   ███╗ ██████╗██████╗ ██████╗  ██████╗  ██████╗
    ████╗ ████║██╔════╝██╔══██╗██╔══██╗██╔═══██╗██╔════╝
    ██╔████╔██║██║     ██████╔╝██║  ██║██║   ██║██║
    ██║╚██╔╝██║██║     ██╔═══╝ ██║  ██║██║   ██║██║
    ██║ ╚═╝ ██║╚██████╗██║     ██████╔╝╚██████╔╝╚██████╗
    ╚═╝     ╚═╝ ╚═════╝╚═╝     ╚═════╝  ╚═════╝  ╚═════╝


Launching MCPDOC server with 2 doc sources
INFO:     Started server process [1919]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8082 (Press CTRL+C to quit)

Run [MCP inspector](https://modelcontextprotocol.io/docs/tools/inspector) and connect to the running server:

```bash
npx @modelcontextprotocol/inspector

Starting MCP inspector...
⚙️ Proxy server listening on port 6277
🔍 MCP Inspector is up and running at http://127.0.0.1:6274 🚀
```

![](https://github.com/user-attachments/assets/14645d57-1b52-4a5e-abfe-8e7756772704)

> * Here, you can test the `Tools` calls.

### Connect to Cursor 

* Open `Cursor Settings` and `MCP` tab.
* This will open the `~/.cursor/mcp.json` file.

![](https://github.com/user-attachments/assets/3d1c8eb3-4d40-487f-8bad-3f9e660f770a)

* Paste the following into the file (we use the `langgraph-docs-mcp` name and link to the LangGraph `llms.txt`).

```json
{
  "mcpServers": {
    "langgraph-docs-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "mcpdoc",
        "mcpdoc",
        "--urls",
        "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt LangChain:https://python.langchain.com/llms.txt",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

* Confirm that the server is running in your `Cursor Settings/MCP` tab.
* Best practice is to then update Cursor Global (User) rules.
* Open Cursor `Settings/Rules` and update `User Rules` with the following (or similar):

```
for ANY question about LangGraph, use the langgraph-docs-mcp server to help answer -- 
+ call list_doc_sources tool to get the available llms.txt file
+ call fetch_docs tool to read it
+ reflect on the urls in llms.txt 
+ reflect on the input question 
+ call fetch_docs on any urls relevant to the question
+ use this to answer the question
```

#### Test in Cursor

* `CMD+L` (on Mac) to open chat.
* Ensure `agent` is selected. 

![](https://github.com/user-attachments/assets/0dd747d0-7ec0-43d2-b6ef-cdcf5a2a30bf)

Then, try an example prompt, such as:
```
what are types of memory in LangGraph?
```

![](https://github.com/user-attachments/assets/180966b5-ab03-4b78-8b5d-bab43f5954ed)
