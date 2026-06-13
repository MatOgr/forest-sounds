import json
import os
import re

# from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
# mcp = FastMCP("Notebook Path Cleaner")

"""
{
    "mcpServers": {
        "notebook-cleaner": {
            "command": "uv",
            "args": [
                    "--directory",
                    "/ABSOLUTE/PATH/TO/YOUR/notebook-cleaner-mcp",
                    "run",
                    "server.py"
            ]
        }
    }
}
"""


# @mcp.tool()
def sanitize_notebook_paths(notebook_path: str, project_root: str) -> str:
    """
    Scans a Jupyter Notebook (.ipynb) file and replaces any instances of absolute local paths
    (e.g., /Users/username/projects/my-project/data/file.csv) in inputs and outputs
    with relative paths starting from the project root.
    """
    notebook_path = os.path.abspath(notebook_path)
    project_root = os.path.abspath(project_root)

    if not os.path.exists(notebook_path):
        return f"Error: Notebook file not found at {notebook_path}"

    # Escaping project root path to safely use it inside regex matches
    escaped_root = re.escape(project_root)
    # Regex to capture absolute path sequences branching into subdirectories
    pattern = re.compile(rf"{escaped_root}(/[^\s\"'\)]*)?")

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        modified_count = 0

        # Helper function to recursively traverse and substitute text inside the JSON structure
        def clean_value(obj):
            nonlocal modified_count
            if isinstance(obj, str):
                new_str, count = pattern.subn(r".\1", obj)  # Prefix with relative '.'
                if count > 0:
                    modified_count += count
                return new_str
            elif isinstance(obj, list):
                return [clean_value(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: clean_value(v) for k, v in obj.items()}
            return obj

        cleaned_nb = clean_value(nb_data)

        if modified_count > 0:
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_nb, f, indent=1)
            return f"Success! Replaced {modified_count} local absolute paths with relative paths."
        else:
            return "No local absolute paths found in cell inputs or outputs."

    except Exception as e:
        return f"An error occurred while processing: {str(e)}"


if __name__ == "__main__":
    # mcp.run(transport="stdio")
