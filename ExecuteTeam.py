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
from function import make_combined_json
from typing import Optional
from config import get_model_client, mapping_dir
import os

async def get_Exeteam(
    task: Optional[str] = None,
    customer_name: Optional[str] = None,
    mapping: Optional[dict] = None,
    field_list: Optional[list] = None
):
    model_client = get_model_client(temperature=0.0)    
    #DSLModifier
    DSLModifier_mem = ListMemory()
    field_expl = field_description(os.path.join(mapping_dir,"filter_field_description.csv"), ["Field","Explanation","Calculation Method and Data Extraction Method"])
    DSLGenmem = make_combined_json(field_explanation = field_expl)
    await DSLModifier_mem.add(MemoryContent(content=DSLGenmem, mime_type=MemoryMimeType.JSON))

    #DSLExecutor
    DSLExecutor_mem = ListMemory()
    # metrics_hint = field_description(field_csv_path, ["Field", "Explanation", "Calculation Method and Data Extraction Method"])
    await DSLExecutor_mem.add(MemoryContent(content=customer_name, mime_type=MemoryMimeType.TEXT))
    #memory end

    DSLModifier = AssistantAgent(
        "DSLModifier",
        description = "Generates a fully executable OpenSearch DSL query by refining DSL Writer’s draft and ensuring flawless JSON syntax.",
        model_client = model_client,
        tools = [],
        memory= [DSLModifier_mem], 
        system_message = """
        You are **DSLModifier** Base on other agent's suggestion, to do the **minimal modifiction** of the dsl.

        Your Output:
        DSL in Excutable JSON format: 
        ```json
        ```
    """
    )
    DSLModifier_cleaned = MessageFilterAgent(
        name="DSLModifier",
        wrapped_agent=DSLModifier,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="DSLExecutor", position="last", count=3),
                    PerSourceFilter(source="user",position="first",count=1)
        ]),
    )

    

    DSLExecutor = AssistantAgent(
        "DSLExecutor",
        description = "Check the metrics of the generated query",
        model_client = model_client,
        tools = [Opendistro_search],
        memory= [DSLExecutor_mem],
        reflect_on_tool_use=True,
        system_message = """
        You are **DSLExecutor** — responsible for executing the provided OpenSearch query and deciding the next step based on the result. When done, say "TASK COMPLETE"
        You are given access to one tool: `opensearch_tool`, which takes the following parameters:
        - customer_name (str), use {memory}.
        - query_str (str)
        - filename (str, optional, default: "result.json") 

        Use the tool properly. 

        Your Goals:
        1. Use `opensearch_tool` to run the query.
        2. base on execution result and give suggestion to modify dsl .
        
        if no Error message, Output:
        ```json
            <File Path>
        ``` \n
        TASK COMPLETE
        else: 
            < suggestion for modification >
            
    """
    )
    
    DSLExecutor_cleaned = MessageFilterAgent(
        name="DSLExecutor",
        wrapped_agent = DSLExecutor,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="DSLModifier", position="last", count=1),
                    PerSourceFilter(source="user",position="first",count=1)
        ]),
    )
    max_msg_termination = MaxMessageTermination(max_messages=5)
    text_termination = TextMentionTermination("TASK COMPLETE")
    

    team = RoundRobinGroupChat(
        [DSLExecutor_cleaned, DSLModifier_cleaned],
        termination_condition = text_termination,
    )

    return team
    

    


    
    