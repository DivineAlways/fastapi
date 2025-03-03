# main.py
import os
import sys
import json
import time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
import openai
from openai import pydantic_function_tool
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Load environment variables (from .env file or platform settings)
load_dotenv()

# Initialize rich console for logging (prints will show in logs)
console = Console()

# Ensure required API keys are set in the environment
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not FIRECRAWL_API_KEY:
    console.print("[red]Error: FIRECRAWL_API_KEY not found in environment variables[/red]")
    sys.exit(1)
if not OPENAI_API_KEY:
    console.print("[red]Error: OPENAI_API_KEY not found in environment variables[/red]")
    sys.exit(1)

# Initialize Firecrawl and OpenAI clients (this uses your working, older version)
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
client = openai.OpenAI()  # Using the version that worked for you

# --- Define Pydantic Models for Function Tools ---
class ScrapeUrlArgs(BaseModel):
    reasoning: str = Field(..., description="Explanation for why we're scraping this URL")
    url: str = Field(..., description="The URL to scrape")
    output_file_path: str = Field(..., description="Path to save the scraped content")

class ReadLocalFileArgs(BaseModel):
    reasoning: str = Field(..., description="Explanation for why we're reading this file")
    file_path: str = Field(..., description="Path of the file to read")

class UpdateLocalFileArgs(BaseModel):
    reasoning: str = Field(..., description="Explanation for why we're updating this file")
    file_path: str = Field(..., description="Path of the file to update")
    content: str = Field(..., description="New content to write to the file")

class CompleteTaskArgs(BaseModel):
    reasoning: str = Field(..., description="Explanation of why the task is complete")

# Create the tools list from our pydantic models
tools = [
    pydantic_function_tool(ScrapeUrlArgs),
    pydantic_function_tool(ReadLocalFileArgs),
    pydantic_function_tool(UpdateLocalFileArgs),
    pydantic_function_tool(CompleteTaskArgs),
]

# --- Agent Prompt Template ---
AGENT_PROMPT = """<purpose>
    You are a world-class web scraping and content filtering expert.
    Your goal is to scrape web content and filter it according to the user's needs.
</purpose>

<instructions>
    <instruction>Run scrape_url, then read_local_file, then update_local_file as many times as needed to satisfy the user's prompt, then complete_task when the user's prompt is fully satisfied.</instruction>
    <instruction>When processing content, extract exactly what the user asked for - no more, no less.</instruction>
    <instruction>When saving processed content, use proper markdown formatting.</instruction>
    <instruction>Use tools available in 'tools' section.</instruction>
</instructions>

<tools>
    <tool>
        <n>scrape_url</n>
        <description>Scrapes content from a URL and saves it to a file</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why we need to scrape this URL</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>url</n>
                <type>string</type>
                <description>The URL to scrape</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>output_file_path</n>
                <type>string</type>
                <description>Where to save the scraped content</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
    
    <tool>
        <n>read_local_file</n>
        <description>Reads content from a local file</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why we need to read this file</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>file_path</n>
                <type>string</type>
                <description>Path of file to read</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
    
    <tool>
        <n>update_local_file</n>
        <description>Updates content in a local file</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why we need to update this file</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>file_path</n>
                <type>string</type>
                <description>Path of file to update</description>
                <required>true</required>
            </parameter>
            <parameter>
                <n>content</n>
                <type>string</type>
                <description>New content to write to the file</description>
                <required>true</required>
            </parameter>
        </parameters>
    </tool>
    
    <tool>
        <n>complete_task</n>
        <description>Signals that the task is complete</description>
        <parameters>
            <parameter>
                <n>reasoning</n>
                <type>string</type>
                <description>Why the task is now complete</description>
                <required>true</required>
            </parameter>
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

# --- Logging Functions (using rich) ---
def log_function_call(function_name: str, function_args: dict):
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in function_args.items())
    console.print(Panel(f"{function_name}({args_str})", title="[blue]Function Call[/blue]", border_style="blue"))

def log_function_result(function_name: str, result: str):
    console.print(Panel(str(result), title=f"[green]{function_name} Result[/green]", border_style="green"))

def log_error(error_msg: str):
    console.print(Panel(str(error_msg), title="[red]Error[/red]", border_style="red"))

# --- Tool Implementations ---
def scrape_url(reasoning: str, url: str, output_file_path: str) -> str:
    log_function_call("scrape_url", {"reasoning": reasoning, "url": url, "output_file_path": output_file_path})
    try:
        response = firecrawl_app.scrape_url(url=url, params={"formats": ["markdown"]})
        if response.get("markdown"):
            content = response["markdown"]
            with open(output_file_path, "w") as f:
                f.write(content)
            log_function_result("scrape_url", f"Successfully scraped {len(content)} characters")
            return content
        else:
            error = response.get("error", "Unknown error")
            log_error(f"Error scraping URL: {error}")
            return ""
    except Exception as e:
        log_error(f"Error scraping URL: {str(e)}")
        return ""

def read_local_file(reasoning: str, file_path: str) -> str:
    log_function_call("read_local_file", {"reasoning": reasoning, "file_path": file_path})
    try:
        console.log(f"[blue]Reading File[/blue] - File: {file_path} - Reasoning: {reasoning}")
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        console.log(f"[red]Error reading file: {str(e)}[/red]")
        return ""

def update_local_file(reasoning: str, file_path: str, content: str) -> str:
    log_function_call("update_local_file", {"reasoning": reasoning, "file_path": file_path, "content": f"{len(content)} characters"})
    try:
        console.log(f"[blue]Updating File[/blue] - File: {file_path} - Reasoning: {reasoning}")
        with open(file_path, "w") as f:
            f.write(content)
        log_function_result("update_local_file", f"Successfully wrote {len(content)} characters")
        return "File updated successfully"
    except Exception as e:
        console.log(f"[red]Error updating file: {str(e)}[/red]")
        return f"Error: {str(e)}"

def complete_task(reasoning: str) -> str:
    log_function_call("complete_task", {"reasoning": reasoning})
    console.log(f"[green]Task Complete[/green] - Reasoning: {reasoning}")
    result = "Task completed successfully"
    log_function_result("complete_task", result)
    return result

# --- Agent Runner Function ---
def run_agent(url: str, prompt: str, output_file_path: str, compute_limit: int):
    # Format the prompt with the provided values
    formatted_prompt = (
        AGENT_PROMPT.replace("{{user_prompt}}", prompt)
                    .replace("{{url}}", url)
                    .replace("{{output_file_path}}", output_file_path)
    )
    messages = [{"role": "user", "content": formatted_prompt}]
    iterations = 0
    break_loop = False

    while iterations < compute_limit:
        if break_loop:
            break
        iterations += 1
        console.rule(f"[yellow]Agent Loop {iterations}/{compute_limit}[/yellow]")
        try:
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
                    console.print(
                        Panel(f"Processing tool call: {function_name}({function_args})",
                              title="[yellow]Tool Call[/yellow]",
                              border_style="yellow")
                    )
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
                        result = f"Error executing {function_name}({function_args}): {str(e)}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(result),
                    })
            else:
                raise ValueError("No tool calls found - should not happen")
        except Exception as e:
            log_error(f"Error: {str(e)}")
            console.print("[yellow]Messages at error:[/yellow]")
    if iterations >= compute_limit:
        log_error("Reached maximum number of iterations")
        raise Exception("Reached maximum number of iterations")
    return messages

# --- FastAPI Setup ---
app = FastAPI()

# Define a Pydantic model for the API request
class AgentRequest(BaseModel):
    url: str
    prompt: str
    output_file_path: str = "scraped_content.md"
    compute_limit: int = 10

@app.post("/run-agent")
def run_agent_endpoint(request: AgentRequest):
    try:
        messages = run_agent(request.url, request.prompt, request.output_file_path, request.compute_limit)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
