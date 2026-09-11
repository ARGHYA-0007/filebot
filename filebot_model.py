from pathlib import Path
import json

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. WORKSPACE
# ============================================================

# Everything the AI creates will be inside this folder.
DESKTOP = Path.home() / "Desktop"
BASE_DIR = DESKTOP / "AI_Workspace"

# Create AI_Workspace if it doesn't exist
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SAFE PATH FUNCTION
# ============================================================

def get_safe_path(file_path: str) -> Path:
    """
    Convert a user-provided path into a safe path inside
    AI_Workspace.

    Prevents things like:
        ../../Windows/System32
        C:/Users/...
    """

    path = (BASE_DIR / file_path).resolve()

    # Make sure the resulting path is inside BASE_DIR
    try:
        path.relative_to(BASE_DIR)
    except ValueError:
        raise ValueError(
            "Access denied: path must stay inside AI_Workspace."
        )

    return path


# ============================================================
# 3. CREATE FOLDER TOOL
# ============================================================

@tool
def create_folder(folder_path: str) -> str:
    """
    Create a folder inside AI_Workspace.

    Example:
        create_folder("myproject")
        create_folder("myproject/src")
    """

    try:
        path = get_safe_path(folder_path)

        path.mkdir(parents=True, exist_ok=True)

        return f"Folder created successfully: {path}"

    except Exception as e:
        return f"Error creating folder: {e}"


# ============================================================
# 4. WRITE FILE TOOL
# ============================================================

@tool
def write_file(file_path: str, content: str) -> str:
    """
    Create or overwrite a text-based file inside AI_Workspace.

    Works with files such as:
        .py
        .cpp
        .c
        .html
        .css
        .js
        .json
        .txt
        .md
        .sql
        .ipynb
        etc.
    """

    try:
        path = get_safe_path(file_path)

        # Create parent folders automatically
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            content,
            encoding="utf-8"
        )

        return f"File written successfully: {path}"

    except Exception as e:
        return f"Error writing file: {e}"


# ============================================================
# 5. READ FILE TOOL
# ============================================================

@tool
def read_file(file_path: str) -> str:
    """
    Read a text file from AI_Workspace.
    """

    try:
        path = get_safe_path(file_path)

        if not path.exists():
            return f"File does not exist: {file_path}"

        if not path.is_file():
            return f"Not a file: {file_path}"

        return path.read_text(
            encoding="utf-8"
        )

    except Exception as e:
        return f"Error reading file: {e}"


# ============================================================
# 6. LIST FILES TOOL
# ============================================================

@tool
def list_files(folder_path: str = "") -> str:
    """
    List files and folders inside AI_Workspace.

    Example:
        list_files("")
        list_files("myproject")
    """

    try:
        path = get_safe_path(folder_path)

        if not path.exists():
            return f"Folder does not exist: {folder_path}"

        items = []

        for item in path.iterdir():

            if item.is_dir():
                items.append(f"[FOLDER] {item.name}")

            else:
                items.append(f"[FILE]   {item.name}")

        if not items:
            return "Folder is empty."

        return "\n".join(items)

    except Exception as e:
        return f"Error listing files: {e}"


# ============================================================
# 7. DELETE FILE TOOL
# ============================================================

@tool
def delete_file(file_path: str) -> str:
    """
    Delete a file inside AI_Workspace.
    """

    try:
        path = get_safe_path(file_path)

        if not path.exists():
            return f"File does not exist: {file_path}"

        if not path.is_file():
            return f"Not a file: {file_path}"

        path.unlink()

        return f"File deleted: {file_path}"

    except Exception as e:
        return f"Error deleting file: {e}"


# ============================================================
# 8. DELETE FOLDER TOOL
# ============================================================

@tool
def delete_folder(folder_path: str) -> str:
    """
    Delete an EMPTY folder inside AI_Workspace.
    """

    try:
        path = get_safe_path(folder_path)

        if not path.exists():
            return f"Folder does not exist: {folder_path}"

        if not path.is_dir():
            return f"Not a folder: {folder_path}"

        path.rmdir()

        return f"Folder deleted: {folder_path}"

    except Exception as e:
        return (
            "Could not delete folder. "
            "Make sure it is empty."
        )


# ============================================================
# 9. CREATE IPYNB TOOL
# ============================================================

@tool
def create_notebook(file_path: str, cells: list[str]) -> str:
    """
    Create a Jupyter Notebook (.ipynb).

    file_path example:
        notebooks/test.ipynb

    cells should be a list of Python code strings.
    """

    try:

        path = get_safe_path(file_path)

        # Make sure extension is ipynb
        if path.suffix.lower() != ".ipynb":
            return "Error: file must have .ipynb extension."

        path.parent.mkdir(parents=True, exist_ok=True)

        notebook_cells = []

        for code in cells:

            notebook_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    line + "\n"
                    for line in code.splitlines()
                ]
            })

        notebook = {
            "cells": notebook_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                notebook,
                f,
                indent=2
            )

        return f"Jupyter notebook created: {path}"

    except Exception as e:
        return f"Error creating notebook: {e}"


# ============================================================
# 10. TOOLS LIST
# ============================================================

tools = [
    create_folder,
    write_file,
    read_file,
    list_files,
    delete_file,
    delete_folder,
    create_notebook
]


# ============================================================
# 11. LLM
# ============================================================

# llm = ChatOllama(
#     model="qwen2.5:7b",
#     temperature=0
# )
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 12. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a filesystem coding assistant.

You have access to a workspace on the user's Desktop:

AI_Workspace

You can create folders and files inside this workspace.

Available operations:

- create folders
- create files
- write files
- read files
- list files
- delete files
- delete empty folders
- create Jupyter notebooks

IMPORTANT RULES:

1. Always use tools when the user asks you to create,
   modify, read, or inspect files.

2. Do not pretend that a file was created.
   Actually call the appropriate tool.

3. All paths are relative to AI_Workspace.

4. Never try to access files outside AI_Workspace.

5. For Python files use .py.

6. For C++ files use .cpp.

7. For C files use .c.

8. For Jupyter notebooks use the create_notebook tool.

9. When creating multiple files, create each required file.

10. If a requested file already exists, explain that
    writing it will overwrite it before doing so when
    the request is ambiguous.

11. After completing an operation, clearly tell the user
    what was created or changed.

The workspace is located at:

""" + str(BASE_DIR)


# ============================================================
# 13. LANGGRAPH NODE
# ============================================================

def chatbot(state: MessagesState):

    messages = state["messages"]

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + messages
    )

    return {
        "messages": [response]
    }


# ============================================================
# 14. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(MessagesState)


builder.add_node(
    "chatbot",
    chatbot
)

builder.add_node(
    "tools",
    ToolNode(tools)
)


builder.add_edge(
    START,
    "chatbot"
)


builder.add_conditional_edges(
    "chatbot",
    tools_condition
)


builder.add_edge(
    "tools",
    "chatbot"
)


workflow = builder.compile()


# ============================================================
# 15. CHAT LOOP
# ============================================================

def main():

    print("=" * 60)
    print("AI FILESYSTEM AGENT")
    print("=" * 60)

    print()

    print(
        f"Workspace:\n{BASE_DIR}"
    )

    print()

    print("Type 'exit' to quit.")
    print()

    messages = []

    while True:

        user_input = input("You: ")

        if user_input.lower() in [
            "exit",
            "quit"
        ]:
            print("Goodbye!")
            break

        messages.append(
            HumanMessage(
                content=user_input
            )
        )

        try:

            result = workflow.invoke(
                {
                    "messages": messages
                }
            )

            # Keep conversation history
            messages = result["messages"]

            response = messages[-1]

            print()
            print("AI:", response.content[0]['text'])
            print()

        except Exception as e:

            print()
            print("ERROR:", e)
            print()


# ============================================================
# 16. START
# ============================================================

if __name__ == "__main__":
    main()