import os, json, re, httpx, asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Gemma-MCP-Gateway")

# Configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("GEMMA_MODEL", "gemma3")

# Official trigger phrase for Gemma 3 function calling
GEMMA_SYSTEM_PROMPT = "You are a model that can do function calling with the following functions"

def format_tools_for_gemma(tools):
    """Format tools using official declaration tokens."""
    definitions = [f"declaration:{t.name}{json.dumps(t.input_schema)}" for t in tools]
    return f"<start_function_declaration>\n" + "\n".join(definitions) + "\n<end_function_declaration>"

def parse_gemma_tool_call(text):
    """Parses official Gemma 3 tool calls, handling <escape> tokens."""
    # Remove <escape> tokens if present before parsing JSON to prevent breakage
    clean_text = text.replace("<escape>", "")
    
    # Official native token pattern
    call_regex = r"<start_function_call>call:(\w+)(\{.*?\})<end_function_call>"
    match = re.search(call_regex, clean_text, re.DOTALL)
    
    if match:
        tool_name = match.group(1)
        try:
            return tool_name, json.loads(match.group(2)), ""
        except:
            return None, None, ""
    return None, None, ""

@mcp.tool()
async def gemma_chat(prompt: str, history: list = None) -> str:
    """
    A tool-augmented chat interface utilizing official Gemma 3 'Native Token' strategies.
    Handles developer role activation and recursive tool execution.
    """
    all_tools = mcp.list_tools()
    available_tools = [t for t in all_tools if t.name != "gemma_chat"]
    
    # Construct the Developer turn (Turn 1) - Official trigger for Tool-use Mode
    tool_block = format_tools_for_gemma(available_tools)
    full_prompt = f"<start_of_turn>developer\n{GEMMA_SYSTEM_PROMPT}{tool_block}<end_of_turn>\n"
    
    # Append conversation history if provided
    if history:
        for turn in history:
            full_prompt += f"<start_of_turn>{turn['role']}\n{turn['content']}<end_of_turn>\n"
            
    # Add final user prompt
    full_prompt += f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    async with httpx.AsyncClient() as client:
        current_prompt = full_prompt
        # Support up to 5 tool call rounds to prevent cycles
        for _ in range(5):
            response = await client.post(
                OLLAMA_URL, 
                json={
                    "model": MODEL_NAME, 
                    "prompt": current_prompt, 
                    "stream": False, 
                    "raw": True # Required for precise control token handling
                }, 
                timeout=120.0
            )
            
            if response.status_code != 200:
                return f"Error from Ollama ({response.status_code}): {response.text}"
                
            output = response.json().get("response", "")
            tool_name, tool_args, _ = parse_gemma_tool_call(output)
            
            if tool_name:
                try:
                    tool_result = await mcp.call_tool(tool_name, tool_args)
                    
                    # Official native response format
                    res_block = f"<start_function_response>{tool_result}<end_function_response>"
                    
                    # Append result back to the prompt as a user turn continuation
                    current_prompt += output + f"\n<start_of_turn>user\n{res_block}<end_of_turn>\n<start_of_turn>model\n"
                except Exception as e:
                    err_block = f"<start_function_response>Error: {str(e)}<end_function_response>"
                    current_prompt += output + f"\n<start_of_turn>user\n{err_block}<end_of_turn>\n<start_of_turn>model\n"
            else:
                # No tool call detected, return final model output
                return output
                
    return "Max tool iterations reached."

@mcp.tool()
async def read_local_file(path: str) -> str:
    """Reads a file from the local filesystem."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
async def get_system_info() -> str:
    """Get basic system information."""
    import platform
    return f"OS: {platform.system()} {platform.release()}, Arch: {platform.machine()}"

if __name__ == "__main__":
    # Single entry point for mcp server lifecycle
    mcp.run()
