# ==========================================================================================================
# Imports
# ==========================================================================================================

import asyncio
import base64
import io
import sys
import json
import gradio as gr
import requests
import time
import uvicorn


from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from mcp_clients.base_http_client import MCPHTTPClient
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from PIL import Image



# ==========================================================================================================
# Application
# ==========================================================================================================

class App():
    """GUI client application that extends the base HTTP client with Gradio interface."""

    def __init__(self, server_url: str, roots_dir: str):
        self.server_url = server_url
        self.roots_dir = roots_dir

        self.tools_cache = []
        self.prompts_cache = []

        self.client = None

        self.llm = ChatOllama(model="qwen3-vl:4b", base_url="http://127.0.0.1:11434")

    # MCP Host — ReAct Agent Loop
    async def chat_with_agent(self, user_message: str, history: list) -> str:
        mcp_tools = await self.client.list_tools()

        # Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]

        model = self.llm.bind_tools(openai_tools)

        # Build the message list from chat history and the new user message
        messages = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        # ReAct loop — call tools until the LLM returns a plain text reply
        for _ in range(10):
            response = await model.ainvoke(messages)
            print(f"MODEL RESPONSE: {response}")
            messages.append(response)

            # No tool calls means the LLM is done — return the final response
            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                return str(raw)

            # Execute each tool call via the MCP server and feed results back
            for tool_call in response.tool_calls:
                print(f"TOOL CALL: \nname: {tool_call["name"]}\n args:{tool_call["args"]}\n")
                result = await self.client.call_tool(tool_call["name"], tool_call["args"])
                print(f"TOOL CALL RESULT: {result}")
                tool_output = " ".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in result.content
                ) if result.content else "(no result)"
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return ("I wasn't able to complete that request. Please try again.")


    async def handle_chat(self, user_message, history):
        if history is None:
            history = []
        if not user_message or not user_message.strip():
            yield history
            return

        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "Thinking..."},
        ]
        yield history

        response_text = await self.chat_with_agent(user_message, history)
        history = history + [{"role": "assistant", "content": response_text}]
        yield history

    async def process_user_request(self, user_message, history, image_input):
        if image_input:
            history = history + [
                {"role": "user", "content": f"The user input the Image: {image_input}."},
            ]

        async for update in self.handle_chat(user_message, history):
            yield update

    async def create_interface(self):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()

        self.tools_cache = await self.client.list_tools()

        with gr.Blocks(title="MCP HTTP Client") as interface:
            gr.Markdown("# MCP HTTP Client - Remote Server Access")
            gr.Markdown(f"""
            **Server:** {self.server_url}
            **Workspace Roots:** {self.roots_dir}

            This client connects to a remote MCP server via HTTP transport.
            All file operations are restricted to the workspace roots directory.
            """)

            with gr.Tabs():
                with gr.Tab("Chat"):
                    chatbot = gr.Chatbot(height=500)
                    msg_input = gr.Textbox(
                        label="Ask about restaurants",
                        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
                    )

                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(
                                label="Dish photo",
                                type="filepath",
                                sources=["upload", "clipboard", "webcam"],
                            )

                    msg_input.submit(
                        self.process_user_request,
                        inputs=[msg_input, chatbot, image_input],
                        outputs=[chatbot],
                    ).then(lambda: ("", None), None, [msg_input, image_input])

        return (interface)



# ==========================================================================================================
# Beginning of execution
# ==========================================================================================================

async def main():
    host_app = App(server_url=sys.argv[1], roots_dir=sys.argv[2])
    interface = await host_app.create_interface()

    app = gr.mount_gradio_app(FastAPI(), interface, path="/")
    config = uvicorn.Config(app, host="127.0.0.1", port=7861, log_level="info")

    try:
        await uvicorn.Server(config).serve() # runs in loop A, doesn't block it
    finally:
        await host_app.client.cleanup()

asyncio.run(main())