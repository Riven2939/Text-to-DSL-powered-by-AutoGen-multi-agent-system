from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from memory import field_description
# from file_path import filter_field_description
import re
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
import json
from tool import Opendistro_search
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from config import get_model_client
from typing import Optional

async def get_ReportSaverTeam(
    task: Optional[str] = None,
    customer_name: Optional[str] = None,
    mapping: Optional[dict] = None,
    field_list: Optional[list] = None
):
    model_client = get_model_client()    
   
    ReportSaver = AssistantAgent(
        "ReportSaver",
        description = "Check the metrics of the generated query",
        model_client = model_client,
        tools = [],
        memory= [],
        reflect_on_tool_use=True,
        system_message = """
            You are a ReportSaver.

            Given an Elasticsearch/OpenSearch DSL query, your task is to extract:

            1. "title": A concise summary of what the DSL query does (≤ 20 words).
            2. "logic": A clear natural language explanation of its logic.
            3. "content": Structured components including:
            - "summary": What the query is analyzing.
            - "metrics": All computed metrics (e.g. max, sum, count).
            - "filters": All filter conditions.
            - "groupings": Aggregation group-by fields.
            - "time_granularity": Time-based interval (e.g. per hour).
            - "methodology": Any special techniques (e.g. serial_diff, bucket_script).

            Respond in this format:
            ```json
            {
                "title": "...",
                "logic": "...",
                "content": {
                    "summary": "...",
                    "metrics": [...],
                    "filters": [...],
                    "groupings": [...],
                    "time_granularity": [...],
                    "methodology": [...]
                }
            }
            ```
    """
    )


    return ReportSaver
    

    


    
    