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

async def get_ModelSelector(
    task: Optional[str] = None,
    customer_name: Optional[str] = None,
    mapping: Optional[dict] = None,
    field_list: Optional[list] = None
):
    model_client = get_model_client()    
   
    ReportSaver = AssistantAgent(
        "ModeSelector",
        description = "Check the metrics of the generated query",
        model_client = model_client,
        tools = [],
        memory= [],
        reflect_on_tool_use=True,
        system_message = """
            You are ModeSelectorAgent.

            ## ROLE
            Your sole job is to read the latest User Requirement (UR), assess task difficulty, and choose the execution mode:
            - "fast": for simple, low-risk, well-specified tasks.
            - "thinking": for complex, ambiguous, or risk-prone tasks that benefit from deeper reasoning.

            
            Output:
                Thinking mode

            OR

            Output:
                Fast mode   
    """
    )


    return ReportSaver