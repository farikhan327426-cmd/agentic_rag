# src/agentic_self_rag/agentic_rag/nodes/router.py

import yaml
from pydantic import BaseModel, Field
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger
from src.agentic_self_rag.core.exceptions import ConfigurationError
from ..state import AgentState

class RouteQuery(BaseModel):
    """Route a user query to the most appropriate datasource."""
    datasource: str = Field(
        ...,
        description="Given a user question choose to route it to direct or vectorstore.",
    )

def route_question(state: AgentState):
    question = state["question"]
    logger.info("--- ROUTING QUESTION ---")

    try:
        # 1. Load Prompt
        with open("config/prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)
        
        # Check if key exists
        if "router_prompts" not in prompts:
            raise ConfigurationError("Missing 'router_prompts' in config/prompts.yaml")
            
        system_prompt = prompts["router_prompts"]["instructions"]

        # 2. Get decision using structured output
        llm = ModelFactory.get_llm(model_type="cheap")
        structured_llm_router = llm.with_structured_output(RouteQuery)

        route = structured_llm_router.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ])

        logger.info(f"--- ROUTING DECISION: {route.datasource} ---")
        return {"route": route.datasource, "question": question}

    except KeyError as e:
        logger.error(f"Configuration KeyError: {e}")
        # Default fallback to vectorstore if config fails
        return {"route": "vectorstore", "question": question}
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        return {"route": "vectorstore", "question": question}