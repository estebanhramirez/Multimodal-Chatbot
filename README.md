# Multimodal-Chatbot
Repository for a Restaurant Chatbot. The GUI is implemented with Gradio. There are two Model Context Protocol (MCP) servers: one provides retrieval functionality; the other provides data management tools, with CRUD priviliges.

## 1. Create the project directory

### 1.1. Create the root folder
Create a folder named `<project_name>`, at some path `~/<prefix_path>` <br>
<br>
**Command `[Command Prompt]`:**
```
mkdir "~/<prefix_path>/<project_name>"
```
**Result `[File directory]`:**
```
<prefix_path>/
        └── <project_name>/
```

### 1.2. Create (local) data folder
Create a folder named `Data`, inside the project folder, `~/<prefix_path>/<project_name>` <br> 
<br>
**Command `[Command Prompt]`:**
```
mkdir "~/<prefix_path>/<project_name>/Data"
```
**Result `[File directory]`:**
```
<prefix_path>/
        └── <project_name>/
                └── Data/
```
Then, move the data to be retrieved to `~/<prefix_path>/<project_name>/Data`.

### 1.2. Clone the GitHub repository
Clone this repository to a folder `Repo`, inside the root folder, `~/<prefix_path>/<project_name>` <br>
<br>
**Command `[Command Prompt]`:**
```
git clone "https://github.com/estebanhramirez/Multimodal-Chatbot.git" "~/<prefix_path>/<project_name>/Repo"
```
**Result `[File directory]`:**
```
<prefix_path>/
        └── <project_name>/
                    ├── Data/
                    └── Repo/
                          ├── README.md
                          └── ...
```

<br>
<br>


## 2. Set-up the virtual environment

### 2.1. Create the virtual environment
Create a new folder name `venv` inside the repository folder, `~/<prefix_path>/<project_name>/Repo`, with an isolated and self-contained virtual environment <br>
<br>
**Command `[Command Prompt]`:**
```
python3.12 -m venv "~/<prefix_path>/<project_name>/Repo/venv"
```

### 2.2. Activate the virtual environment
**Command `[Command Prompt]`:**
```
source "~/<prefix_path>/<project_name>/Repo/venv/bin/activate"
```
or (in Windows)
```
"~\<prefix_path>\<project_name>\Repo\venv\Scripts\activate.bat"
```
We can deactivate the virtual environment using `deactivate.bat`. Please notice that the quotes are important in the path.

<br>
<br>


## 3. Install the required libraries
**Command `[Command Prompt]`:**
```
pip install -r "~/<prefix_path>/<project_name>/Repo/requirements.txt"
```

<br>
<br>


## 4. Set-up OLlama API

### 4.1. Install OLlama
Install OLlama in your local laptop<br>
<br>
**Command `[PowerShell]`:**
```
irm https://ollama.com/install.ps1 | iex
```

### 4.2. Pull Ollama models
Download the model `qwen3-vl:4b`<br>
<br>
**Command `[Command Prompt | PowerShell]`:**
```
ollama pull qwen3-vl:4b
```
Check that the model was downloaded<br>
List the models downloaded locally:<br>
<br>
**Command `[Command Prompt | PowerShell]`:**
```
ollama list
```

### 4.3. Start the Ollama HTTP Server/API
Launch the ollama app to run the LLMs locally:<br>
<br>
**Command `[Command Prompt | PowerShell]`:**
```
ollama serve
```
Ollama runs on `port 11434` by default (bound to `127.0.0.1:11434` / `http://localhost:11434`).

<br>
<br>


## 5. Run this project
### (Optional) 5.0. Populate the vector store
Run the script to populate the chromadb vector store:<br>
<br>
**Command `[Command Prompt]`:**
```
python "~/<prefix_path>/<project_name>/Repo/create_persistent_database.py"
```

### 5.1. Launch the (MCP) retrieval Server
Start the retrieval server, listening on `port 8080` (bound to `127.0.0.1:8080` / `http://localhost:8080`)<br>
<br>
**Command `[Command Prompt]`:**
```
python "~/<prefix_path>/<project_name>/Repo/mcp_servers/retrieval_server.py"
```

### 5.2. Run the (MCP) Host application
Run the script `host_app.py`:<br>
<br>
**Command `[Command Prompt]`:**
```
python "~/<prefix_path>/<project_name>/Repo/host_app.py"
```
