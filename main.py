"""
Improved Firecrawl & OpenAI Agent with In-Memory File Storage
Dependencies:
    - fastapi
    - uvicorn
    - rich
    - openai>=1.63.0
    - pydantic>=2.0.0
    - firecrawl-py>=0.1.0
    - python-dotenv>=1.0.0
Usage (locally):
    uvicorn improved_agent:app --reload
Environment:
    Set FIRECRAWL_API_KEY and OPENAI_API_KEY in your environment or .env file.
"""

import os
import sys
import json
import time
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
import openai
from openai import pydantic_function_tool
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app and rich console for logging
app = FastAPI(title="Improved Firecrawl Agent")
console = Console()

# Get API keys from environment
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not FIRECRAWL_API_KEY or not OPENAI_API_KEY:
    console.print("[red]Error: Both FIRECRAWL_API_KEY and OPENAI_API_KEY must be set[/red]")
    sys.exit(1)

# Initialize external API clients (using your known working versions)
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
client = openai.OpenAI()  # Using the client that you know works

# In-memory file storage (our “virtual file system”)
files_in_memory: Dict[str, str] = {}

##############################################################################
# Define Pydantic Models for Tool Arguments
##############################################################################
class ScrapeUrlArgs(BaseModel):
    reasoning: str = Field(..., description="Why we are scraping this URL")
    url: str = Field(..., description="The URL to scrape")
    output_file_path: str = Field(..., description="Key to save scraped content in memory")

class ReadLocalFileArgs(BaseModel):
    reasoning: str = Field(..., description="Why we need to read this file")
    file_path: str = Field(..., description="Key of the file to read from memory")

class UpdateLocalFileArgs(BaseModel):
    reasoning: str = Field(..., description="Why we are updating this file")
    file_path: str = Field(..., description="Key of the file to update")
    content: str = Field(..., description="New content to write")

class CompleteTaskArgs(BaseModel):
    reasoning: str = Field(..., description="Why the task is complete")

tools = [
    pydantic_function_tool(ScrapeUrlArgs),
    pydantic_function_tool(ReadLocalFileArgs),
    pydantic_function_tool(UpdateLocalFileArgs),
    pydantic_function_tool(CompleteTaskArgs),
]

##############################################################################
# Agent Prompt Template (Instructions for the Agent)
##############################################################################
AGENT_PROMPT = """<purpose>
    You are a world-class web scraping and content filtering expert.
    Your goal is to scrape web content and filter it according to the user's needs.
</purpose>

<instructions>
    <instruction>Run scrape_url, then read_local_file, then update_local_file as needed, and finally complete_task when done.</instruction>
    <instruction>Extract exactly what the user requested—nothing more, nothing less.</instruction>
    <instruction>Format processed content as proper markdown.</instruction>
    <instruction>Use the tools defined in the 'tools' section.</instruction>
</instructions>

<tools>
    <tool>
        <n>scrape_url</n>
        <description>Scrapes content from a URL and saves it in memory</description>
        <parameters>
            <parameter><n>reasoning</n><type>string</type><description>Why we need to scrape this URL</description><required>true</required></parameter>
            <parameter><n>url</n><type>string</type><description>The URL to scrape</description><required>true</required></parameter>
            <parameter><n>output_file_path</n><type>string</type><description>Key for saving the scraped content</description><required>true</required></parameter>
        </parameters>
    </tool>
    <tool>
        <n>read_local_file</n>
        <description>Reads content from an in-memory file</description>
        <parameters>
            <parameter><n>reasoning</n><type>string</type><description>Why we need to read this file</description><required>true</required></parameter>
            <parameter><n>file_path</n><type>string</type><description>Key of the file to read</description><required>true</required></parameter>
        </parameters>
    </tool>
    <tool>
        <n>update_local_file</n>
        <description>Updates content in an in-memory file</description>
        <parameters>
            <parameter><n>reasoning</n><type>string</type><description>Why we need to update this file</description><required>true</required></parameter>
            <parameter><n>file_path</n><type>string</type><description>Key of the file to update</description><required>true</required></parameter>
            <parameter><n>content</n><type>string</type><description>New content to write</description><required>true</required></parameter>
        </parameters>
    </tool>
    <tool>
        <n>complete_task</n>
        <description>Signals that the task is complete</description>
        <parameters>
            <parameter><n>reasoning</n><type>string</type><description>Why the task is complete</description><required>true</required></parameter>
        </parameters>
    </tool>
</tools>

<user-prompt>
    {{user_prompt}}
</user-prompt>

<url>
    {{url}}
</url>

<output-file-path>
    {{output_file_path}}
</output-file-path>
"""

##############################################################################
# Logging Helpers using rich
##############################################################################
def log_function_call(function_name: str, function_args: dict):
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in function_args.items())
    console.print(Panel(f"{function_name}({args_str})", title="[blue]Function Call[/blue]", border_style="blue"))

def log_function_result(function_name: str, result: str):
    console.print(Panel(str(result), title=f"[green]{function_name} Result[/green]", border_style="green"))

def log_error(error_msg: str):
    console.print(Panel(str(error_msg), title="[red]Error[/red]", border_style="red"))

##############################################################################
# Tool Implementations (Working In-Memory)
##############################################################################
def scrape_url(reasoning: str, url: str, output_file_path: str) -> str:
    log_function_call("scrape_url", {"reasoning": reasoning, "url": url, "output_file_path": output_file_path})
    try:
        response = firecrawl_app.scrape_url(url=url, params={"formats": ["markdown"]})
        if "markdown" in response:
            content = response["markdown"]
            limited_content = content[:2000]  # Limit to first 2000 characters
            files_in_memory[output_file_path] = limited_content
            log_function_result("scrape_url", f"Scraped {len(limited_content)} characters (limited)")
            return limited_content
        else:
            error = response.get("error", "Unknown error")
            log_error(f"Error scraping URL: {error}")
            return ""
    except Exception as e:
        log_error(f"Error scraping URL: {str(e)}")
        return ""

def read_local_file(reasoning: str, file_path: str) -> str:
    log_function_call("read_local_file", {"reasoning": reasoning, "file_path": file_path})
    content = files_in_memory.get(file_path, "")
    log_function_result("read_local_file", f"Read {len(content)} characters")
    return content

def update_local_file(reasoning: str, file_path: str, content: str) -> str:
    log_function_call("update_local_file", {"reasoning": reasoning, "file_path": file_path, "content_length": len(content)})
    files_in_memory[file_path] = content
    log_function_result("update_local_file", f"Updated file with {len(content)} characters")
    return "File updated successfully"

def complete_task(reasoning: str) -> str:
    log_function_call("complete_task", {"reasoning": reasoning})
    result = "Task completed successfully"
    log_function_result("complete_task", result)
    return result

##############################################################################
# Agent Runner Function
##############################################################################
def run_agent(url: str, prompt: str, output_file_path: str, compute_limit: int):
    # Clear previous in-memory files
    files_in_memory.clear()
    # Replace placeholders in our agent prompt
    formatted_prompt = (
        AGENT_PROMPT.replace("{{user_prompt}}", prompt)
                    .replace("{{url}}", url)
                    .replace("{{output_file_path}}", output_file_path)
    )
    messages = [{"role": "user", "content": formatted_prompt}]
    iterations = 0
    break_loop = False

    while iterations < compute_limit and not break_loop:
        iterations += 1
        console.rule(f"[yellow]Agent Loop {iterations}/{compute_limit}[/yellow]")
        try:
            # Call the OpenAI API (using your working client)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            response_message = completion.choices[0].message
            assistant_content = response_message.content or ""
            if assistant_content:
                console.print(Panel(assistant_content, title="Assistant"))
            messages.append({"role": "assistant", "content": assistant_content})
            
            if response_message.tool_calls:
                # Process tool calls from the response
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in response_message.tool_calls
                    ],
                })
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    console.print(Panel(f"Processing tool call: {function_name}({function_args})",
                                          title="[yellow]Tool Call[/yellow]", border_style="yellow"))
                    result = None
                    try:
                        if function_name == "ScrapeUrlArgs":
                            result = scrape_url(**function_args)
                        elif function_name == "ReadLocalFileArgs":
                            result = read_local_file(**function_args)
                        elif function_name == "UpdateLocalFileArgs":
                            result = update_local_file(**function_args)
                        elif function_name == "CompleteTaskArgs":
                            result = complete_task(**function_args)
                            break_loop = True
                        else:
                            raise ValueError(f"Unknown function: {function_name}")
                    except Exception as e:
                        error_msg = f"Error executing {function_name}: {str(e)}"
                        console.print(Panel(error_msg, title="[red]Error[/red]"))
                        result = error_msg
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(result),
                    })
            else:
                raise ValueError("No tool calls found - should not happen")
        except Exception as e:
            log_error(f"Agent loop error: {str(e)}")
            break

    return {
        "messages": messages,
        "files_in_memory": files_in_memory
    }

##############################################################################
# FastAPI Request Model and Endpoint
##############################################################################
class AgentRequest(BaseModel):
    url: str
    prompt: str
    output_file_path: str = "scraped_content.md"
    compute_limit: int = 10

@app.post("/run-agent")
def run_agent_endpoint(request: AgentRequest):
    try:
        result = run_agent(
            url=request.url,
            prompt=request.prompt,
            output_file_path=request.output_file_path,
            compute_limit=request.compute_limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Hello from the Improved Firecrawl Agent!"}
