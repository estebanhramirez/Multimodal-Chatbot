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
Download the model `llama2`<br>
<br>
**Command `[Command Prompt | PowerShell]`:**
```
ollama pull qwen3-vl:4b
```
Check that the model was downloaded<br>
<br>
**Command `[Command Prompt | PowerShell]`:**
```
ollama list
```

### 4.3. Start the Ollama HTTP Server/API
**Command `[Command Prompt | PowerShell]`:**
```
ollama serve
```
Ollama runs on `port 11434` by default (bound to `127.0.0.1:11434` / `http://localhost:11434`).

<br>
<br>


## 5. Run this project
## Launch the retrieval server
Start the retrieval server, listening on `port 8080` (bound to `127.0.0.1:8080` / `http://localhost:8080`)<br>
<br>
**Command `[Command Prompt]`:**
```
python "~/<prefix_path>/<project_name>/Repo/mcp_servers/retrieval_server.py"
```
**Result `[Command Prompt]`:**
```
✓ Connected to MCP server

=== MCP Client ===
Commands: tools | call | resources | read | prompts | prompt | quit

> tools                                                                       
  • retrieve_articles: Function to retrieve similar articles from the article database based on a text query,
with optional metadata filtering.
Args:
    chromadb_dir: str
        The directory where the Chroma database is persisted.
    query: str
        The text query to search for.
    k: int
        The number of top results to return.
    where: dict | None
        Optional metadata filtering conditions. For example:
            where = {"cuisine": "Italian", "location": "New York"}
        If None, no filtering is applied.
Returns:
    Four lists: ids, docs, metas, dists
        - ids: List of document IDs of the top-k results.
        - docs: List of document contents of the top-k results.
        - metas: List of metadata dictionaries of the top-k results.
        - dists: List of distances (similarity scores) of the top-k results.
  • initialize_persistent_restaurant_database: None
> call
  Tool name: retrieve_articles
  Arguments (as JSON, for example, {"text": "hello"}): 
  {"chromadb_dir": "chroma_multimodal", "query": "Cozy Mediterranean", "k": 3}
  Result: [["rest_15","rest_134","rest_130"],["Restaurant: The Blue Fig\nCuisine: Eastern Mediterranean\nLocation: Los Feliz","Restaurant: The Marble Arch\nCuisine: Mediterranean\nLocation: Laguna Beach","Restaurant: The Velvet Olive\nCuisine: Modern Italian\nLocation: Los Gatos"],[{"source":"restaurant","location":"Los Feliz","cuisine":"Eastern Mediterranean","doc_id":"rest_15"},{"source":"restaurant","doc_id":"rest_134","location":"Laguna Beach","cuisine":"Mediterranean"},{"source":"restaurant","cuisine":"Modern Italian","doc_id":"rest_130","location":"Los Gatos"}],[0.9135939478874207,0.9808933138847351,0.9998288750648499]]
```
