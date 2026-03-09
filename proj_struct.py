import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

project_name = "agentic_self_rag"

list_of_files = [

    ".env",
    ".gitignore",
    "docker-compose.yaml",
    "Dockerfile",
    "pyproject.toml",
    "README.md",

    "config/settings.yaml",
    "config/prompts.yaml",

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/main.py",

    f"src/{project_name}/core/__init__.py",
    f"src/{project_name}/core/config_loader.py",
    f"src/{project_name}/core/logger.py",
    f"src/{project_name}/core/exceptions.py",

    f"src/{project_name}/database/__init__.py",
    f"src/{project_name}/database/connection.py",
    f"src/{project_name}/database/vector_store.py",

    f"src/{project_name}/ingestion/__init__.py",
    f"src/{project_name}/ingestion/processor.py",
    f"src/{project_name}/ingestion/embedder.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/llm_factory.py",

    f"src/{project_name}/agentic_rag/__init__.py",
    f"src/{project_name}/agentic_rag/graph.py",
    f"src/{project_name}/agentic_rag/state.py",
    f"src/{project_name}/agentic_rag/edges.py",

    f"src/{project_name}/agentic_rag/nodes/__init__.py",
    f"src/{project_name}/agentic_rag/nodes/router.py",
    f"src/{project_name}/agentic_rag/nodes/retrieve.py",
    f"src/{project_name}/agentic_rag/nodes/generate.py",
    f"src/{project_name}/agentic_rag/nodes/graders.py",
    f"src/{project_name}/agentic_rag/nodes/rewriter.py",
    f"src/{project_name}/agentic_rag/nodes/reviser.py",

    f"src/{project_name}/tools/__init__.py",
    f"src/{project_name}/tools/search_tool.py",
    f"src/{project_name}/tools/custom_tools.py",

    "tests/test_nodes.py",
    "tests/test_retrieval.py",

    "data/raw/.gitkeep",
    "research/experiments.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists():
        filepath.touch()
        logging.info(f"Created file: {filepath}")