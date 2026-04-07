# Gemma 3 & FunctionGemma MCP Gateway

This repository provides an official implementation of a bridge between Google's **Gemma 3 / FunctionGemma** and the **Model Context Protocol (MCP)**.

## 🚀 Installation

Ensure you have a modern Python environment (3.10+) and run:

```bash
# Clone the repository and navigate to the example
cd gemma/examples/gemma3-mcp-agent/

# Install the required MCP and communication libraries
pip install -r requirements.txt
```

### Prerequisites
1. **Ollama**: Download from [ollama.com](https://ollama.com).
2. **Gemma 3 Model**: Run `ollama pull gemma3`.

---

## 🏗️ The "Gemma 3 Bridge"

This bridge is uniquely designed to bypass the common "regex parser" failures found in standard implementations. It utilizes the **Official Native Tokens** for high-reliability tool execution:

* **Official Specification**: Aligns with `FunctionGemma` standards using the `declaration:tool_name{schema}` format.
* **Native Transitions**: Uses official control tokens:
    * `<start_function_call>` and `<end_function_call>`
    * `<start_function_response>` and `<end_function_response>`
* **Developer-Role Implementation**: Automatically injects the `developer` turn required to trigger Gemma 3's high-reasoning tool-use mode.
* **Escape Handling**: Built-in support for the `<escape>` token, ensuring JSON inputs remain valid even with complex special characters.

---

## 🧪 Usage & Quick Start

### 1. Using the MCP Inspector (Verification)
To verify the bridge and inspect tool schemas without an IDE, use the `mcp-inspector`:

```bash
npx @modelcontextprotocol/inspector python server.py
```
* Once the inspector loads, you can view the `gemma_chat` tool.
* You can trigger a test call to `get_system_info` or `read_local_file` to see the native token encapsulation in action.

### 2. Integration with Antigravity IDE
1. Open **Antigravity Settings**.
2. Navigate to **MCP Servers**.
3. Import the `mcp_config.json` provided in this directory.
    * *Note: Ensure the `args` path in `mcp_config.json` correctly points to `server.py` relative to your workspace root.*
4. The IDE agent will now be able to use Gemma 3 via the `gemma_chat` tool for local reasoning.

### 3. Verification Test Case
Ask the agent: 
> "Check my system OS and read the content of requirements.txt."

This will trigger a multi-turn reasoning loop:
1. Model generates `<start_function_call>call:get_system_info{}<end_function_call>`.
2. Gateway executes local check and returns `<start_function_response>`.
3. Model generates the second call for `read_local_file`.
