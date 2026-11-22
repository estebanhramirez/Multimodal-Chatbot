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


from mcp_clients.base_http_client import MCPHTTPClient
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from PIL import Image



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

def encode_image(path: str, max_side: int = 512) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))          # downscale; CLIP resizes to 224 anyway
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unwrap_response(response):
    """
    Function to unwrap the response from the Ollama server and extract the content from the streamed JSON lines.

    Args:
        response (requests.Response): The response object from the Ollama server.
    Returns:
        str: The final response content extracted from the streamed JSON lines.
    """

    if response.status_code == 200:
        final_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        content = json_data["message"]["content"]
                        final_response += content
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from line: {line}")
        return(final_response)
    else:
        return(f"Request failed with status code: {response.status_code}")


# Textual LLM
def call_llm(user_prompt: str) -> str:
    """
        Function to send a model execution request to the Ollama server running locally through port 11434,
        and receive and stream the response from the model.
    
        Args:
            system_msg (str): The system message to provide context for the LLM.
            user_prompt (str): The user prompt containing the task and input data.
    
        Returns:
            str: The final response from the LLM after processing the input.
    """

    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model": "phi4-mini:3.8b",
        "messages": [ 
            {"role": "user", "content": user_prompt}
            ]
    }

    response = requests.post(url, json=payload, stream=True)

    return unwrap_response(response)


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

        self.client = None # For lazy initialization

    async def gui_list_tools(self):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()

        tools = await self.client.list_tools()
        self.tools_cache = [(t.name, f"{t.name}: {t.description}") for t in tools]
        output = "\n".join([f"• {t.name}: {t.description}" for t in tools])
        choices = [t.name for t in tools]
        return output, gr.update(choices=choices)

    async def gui_call_tool(self, tool_name, arguments_json):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()

        if not tool_name:
            return "Error: Please select a tool from the dropdown first"
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            result = await self.client.call_tool(tool_name, args)
            output = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    output += content.text + "\n"
            return output if output else "No response"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {e}"

    async def gui_list_resources(self):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()
        
        resources = await self.client.list_resources()
        if resources:
            output = []
            for r in resources:
                name = getattr(r, 'name', getattr(r, 'description', 'Unnamed resource'))
                uri_template = getattr(r, 'uriTemplate', getattr(r, 'uri', 'N/A'))
                output.append(f"• {name}\n  URI template: {uri_template}")
            return "\n\n".join(output)
        return "No resources available"

    async def gui_read_resource(self, uri):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()
        
        if not uri:
            return "Error: Please enter a resource URI"
        try:
            result = await self.client.read_resource(uri)
            output = ""
            for content in result.contents:
                if hasattr(content, 'text'):
                    output += content.text + "\n"
            return output if output else "No content"
        except Exception as e:
            return f"Error: {e}"

    async def gui_list_prompts(self):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()

        prompts = await self.client.list_prompts()
        self.prompts_cache = prompts
        output = []
        choices = []
        for p in prompts:
            args_info = ""
            if p.arguments:
                arg_names = [arg.name for arg in p.arguments]
                args_info = f" (args: {', '.join(arg_names)})"
            output.append(f"• {p.name}: {p.description}{args_info}")
            choices.append(p.name)
        return "\n".join(output), gr.update(choices=choices)

    async def gui_get_prompt(self, prompt_name, arguments_json):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()

        if not prompt_name:
            return "Error: Please select a prompt from the dropdown first"
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            result = await self.client.get_prompt(prompt_name, args)
            output = f"--- Prompt: {result.description} ---\n\n"
            for msg in result.messages:
                content_text = msg.content.text if hasattr(msg.content, 'text') else msg.content.get('text', '')
                output += f"{msg.role}: {content_text}\n"
            return output
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {e}"

    async def get_recipe_from_photo(self, image_input):
        self.client = MCPHTTPClient(server_url=self.server_url, roots_dir=self.roots_dir)
        await self.client.connect()

        args = {"chromadb_dir": "../chroma_multimodal", "query_image": encode_image(image_input), "k": 3}
        chromadb_result = await self.client.call_tool(
            "retrieve_images_by_image",
            args
        )

        output_recipe = ""
        for content in chromadb_result.content:
            if hasattr(content, 'text'):
                output_recipe += content.text + "\n"

        user_prompt = f"""
            Provide step-by-step instructions to complete the following recipe: {output_recipe}
        """
        llm_result = call_llm(user_prompt=user_prompt)

        return (llm_result)


    def create_interface(self):
        """Create the Gradio interface."""

        with gr.Blocks(title="MCP HTTP Client") as interface:
            gr.Markdown("# MCP HTTP Client - Remote Server Access")
            gr.Markdown(f"""
            **Server:** {self.server_url}
            **Workspace Roots:** {self.roots_dir}

            This client connects to a remote MCP server via HTTP transport.
            All file operations are restricted to the workspace roots directory.
            """)

            with gr.Tabs():
                with gr.Tab("Multimodal Chatbot"):
                    gr.Markdown("Upload a photo of a dish to get a step-by-step recipe.")
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(
                                label="Dish photo",
                                type="filepath",          # handler receives a path string (or None)
                                sources=["upload", "clipboard", "webcam"],
                            )
                            call_tool_btn = gr.Button("Send image", variant="primary")
                        with gr.Column():
                            tool_result = gr.Textbox(label="Step-by-step recipe", lines=12)
                    call_tool_btn.click(
                        fn=self.get_recipe_from_photo,
                        inputs=[image_input],
                        outputs=tool_result,
                    )
                with gr.Tab("Agentic workflow"):
                    gr.Markdown("Instructions to replicate your image.")
                    with gr.Row():
                        with gr.Column():
                            tool_args = gr.Textbox(
                                label="Arguments (JSON)",
                                placeholder='{"filepath": "test.txt"}',
                                lines=3
                            )
                            call_tool_btn = gr.Button("Send image", variant="primary")
                            tool_result = gr.Textbox(label="Step-by-step recipe", lines=8)
                    call_tool_btn.click(
                        fn=self.get_recipe_from_photo,
                        inputs=[tool_args],
                        outputs=tool_result
                    )
                with gr.Tab("Tools"):
                    gr.Markdown("### Discover and Execute Server Tools")
                    with gr.Row():
                        with gr.Column():
                            list_tools_btn = gr.Button("List Tools", variant="primary")
                            tools_output = gr.Textbox(label="Available Tools", lines=5)

                        with gr.Column():
                            tool_dropdown = gr.Dropdown(label="Select Tool", choices=[], interactive=True)
                            tool_args = gr.Textbox(
                                label="Arguments (JSON)",
                                placeholder='{"filepath": "test.txt"}',
                                lines=3
                            )
                            call_tool_btn = gr.Button("Call Tool", variant="primary")
                            tool_result = gr.Textbox(label="Tool Result", lines=8)

                    list_tools_btn.click(
                        fn=self.gui_list_tools,
                        outputs=[tools_output, tool_dropdown]
                    )

                    call_tool_btn.click(
                        fn=self.gui_call_tool,
                        inputs=[tool_dropdown, tool_args],
                        outputs=tool_result
                    )

                with gr.Tab("Resources"):
                    gr.Markdown("### Access Server Resources")
                    with gr.Row():
                        with gr.Column():
                            list_resources_btn = gr.Button("List Resource Templates", variant="primary")
                            resources_output = gr.Textbox(label="Available Resources", lines=5)

                        with gr.Column():
                            resource_uri = gr.Textbox(
                                label="Resource URI",
                                placeholder="file://workspace/README.md",
                                lines=1
                            )
                            read_resource_btn = gr.Button("Read Resource", variant="primary")
                            resource_content = gr.Textbox(label="Resource Content", lines=10)

                    list_resources_btn.click(
                        fn=self.gui_list_resources,
                        outputs=resources_output
                    )

                    read_resource_btn.click(
                        fn=self.gui_read_resource,
                        inputs=resource_uri,
                        outputs=resource_content
                    )

                with gr.Tab("Prompts"):
                    gr.Markdown("### List and Get Prompts")
                    with gr.Row():
                        with gr.Column():
                            list_prompts_btn = gr.Button("List Prompts", variant="primary")
                            prompts_output = gr.Textbox(label="Available Prompts", lines=5)

                        with gr.Column():
                            prompt_dropdown = gr.Dropdown(label="Select Prompt", choices=[], interactive=True)
                            prompt_args = gr.Textbox(
                                label="Arguments (JSON)",
                                placeholder='{"filename": "example.py"}',
                                lines=2
                            )
                            get_prompt_btn = gr.Button("Get Prompt", variant="primary")
                            prompt_result = gr.Textbox(label="Prompt Messages", lines=10)

                    list_prompts_btn.click(
                        fn=self.gui_list_prompts,
                        outputs=[prompts_output, prompt_dropdown]
                    )

                    get_prompt_btn.click(
                        fn=self.gui_get_prompt,
                        inputs=[prompt_dropdown, prompt_args],
                        outputs=prompt_result
                    )

        return interface



# ==========================================================================================================
# Beginning of execution
# ==========================================================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python mcp_http_client_app.py <server_url> <roots_dir>")
        print("Example: python mcp_http_client_app.py http://127.0.0.1:8080 /path/to/workspace")
        sys.exit(1)

    server_url = sys.argv[1]
    roots_dir = sys.argv[2]

    client = App(server_url, roots_dir)
    interface = client.create_interface()
    interface.queue().launch(server_name="127.0.0.1", server_port=7861)



# ==========================================================================================================
# Beginning of execution
# ==========================================================================================================

"""
async def main():
    b64 = encode_image("../Data/query_image.jpeg")

    async with streamablehttp_client("http://localhost:8080/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "retrieve_images_by_image",
                {
                    "chromadb_dir": "../chroma_multimodal",
                    "query_image": b64,
                    "k": 2,
                    #"where": {"cuisine": "Italian"},   # optional
                },
            )

            # Newer SDK versions populate structuredContent; otherwise parse the text blocks
            print(result.structuredContent or [c.text for c in result.content])
"""

def main():
    if len(sys.argv) < 3:
        print("Usage: python mcp_http_client_app.py <server_url> <roots_dir>")
        print("Example: python host_app.py http://127.0.0.1:8080 ./")
        sys.exit(1)

    server_url = sys.argv[1]
    roots_dir = sys.argv[2]

    client = App(server_url, roots_dir)
    interface = client.create_interface()
    interface.queue().launch(server_name="127.0.0.1", server_port=7861)



asyncio.run(main())