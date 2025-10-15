from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from memory import field_description
# from file_path import filter_field_description
from autogen_agentchat.agents import AssistantAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
import json
from tool import Opendistro_search
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat,SelectorGroupChat
from autogen_agentchat.ui import Console
from function import make_combined_json
from config import get_model_client, mapping_dir
import os

filter_csv = os.path.join(mapping_dir, "filter_field_description.csv")
print(f"filter_csv: {filter_csv}")

async def get_DSLteam(task, customer_name,mapping,field_list):
    model_client = get_model_client(temperature=0.0)

    #DSLGenerator
    DSLGenerator_mem = ListMemory()
    field_expl = field_description(filter_csv, ["Field","Explanation","Calculation Method and Data Extraction Method"])
    DSLGenmem = make_combined_json(field_explanation = field_expl, Confirmed_FieldList = field_list)
    await DSLGenerator_mem.add(MemoryContent(content=DSLGenmem, mime_type=MemoryMimeType.JSON))

    #MetricsAgent
    MetricsAgent_mem = ListMemory()
    metrics_hint = field_description(filter_csv, ["Field", "Calculation Method and Data Extraction Method"])
    Metricsmem = make_combined_json(field_explanation = metrics_hint, user_requirement = task)
    await MetricsAgent_mem.add(MemoryContent(content=Metricsmem, mime_type=MemoryMimeType.JSON))

    #KeywordChecker
    KeywordChecker_mem = ListMemory()
    # metrics_hint = field_description(field_csv_path, ["Field", "Explanation", "Calculation Method and Data Extraction Method"])
    await KeywordChecker_mem.add(MemoryContent(content=mapping, mime_type=MemoryMimeType.TEXT))
    #memory end

    

    DSLGenerator = AssistantAgent(
        "DSLGenerator",
        description = "Generates a fully executable OpenSearch DSL query by refining DSL Writer’s draft and ensuring flawless JSON syntax.",
        model_client = model_client,
        tools = [],
        memory= [DSLGenerator_mem], 
        system_message = """
        You are **DSLGenerator** — the agent that turns approved requirements into a complete, syntactically-correct OpenSearch DSL query.
        
        You will be provided {memory}:
            • Confirmed_FieldList that are suggested to be used 
            • field_explanation for reference
        Your Goal:
        1. **Understand carefully** user requirement 
        2. Ensure DSL logic match the user requirement 
        3. produce a complete OpenSearch DSL query
        
        NOTE: 
        1. Set the size:0 if user not mentioned. 
        2. Use Hong Kong time as default. 
        3. Do Not filter by "customer" 
        Output content:
        ```json
            < valid JSON DSL for Elastic search >
        ```
    """
    )
    DSLGenerator_cleaned = MessageFilterAgent(
        name="DSLGenerator",
        wrapped_agent=DSLGenerator,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="user", position="last", count=1),
        ]),
    )

    MetricsAgent = AssistantAgent(
        "MetricsAgent",
        description = "Check the metrics of the generated query",
        model_client = model_client,
        tools = [],
        memory= [MetricsAgent_mem], 
        system_message = """
        You are **MetricsAgent** — only check and fix **metrics** in DSL queries.
        DO NOT make unecessary changes if you didn't find any problem. 
        
        You will be provided {memory}:
            • field_explanation for reference

        check the calculation method of metric with following : 
        1.max_bucket, `avg_bucket` if an additional layer of summarization is needed.
        2.make sure the DSL match user requirement

        Output content:
        ```json
            < valid JSON DSL for Elastic search >
        ```
    """
    )
    MetricsAgent_cleaned = MessageFilterAgent(
        name="MetricsAgent",
        wrapped_agent=MetricsAgent,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="DSLGenerator", position="last", count=1),
                    PerSourceFilter(source="user", position="first", count=1),
        ]),
    )


    KeywordChecker = AssistantAgent(
        "KeywordChecker",
        description = "Check the metrics of the generated query",
        model_client = model_client,
        tools = [],
        memory= [KeywordChecker_mem], 
        system_message = """
        You are **KeywordChecker** — You ONLY check and fix **.keyword field** and **time format** in DSL queries.
        
        You will be provided {memory}:
            • index_mapping 
        Your Goal:
            1.Use index_mapping to check DSL fields and ensure all keyword fields have the ".keyword" suffix.
            2.fix the format of @timestamp as: yyyy-MM-dd'T'HH:mm:ss, do not modify the time range.
 
        Output:
        ```json
            < valid JSON DSL for Elastic search >
        ```
    """
    )
    KeywordChecker_cleaned = MessageFilterAgent(
        name="KeywordChecker",
        wrapped_agent=KeywordChecker,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="MetricsAgent", position="last", count=1),
                    # PerSourceFilter(source="user", position="first", count=1),
        ]),
    )
    

    max_msg_termination = MaxMessageTermination(max_messages=4)
    text_termination = TextMentionTermination("TASK COMPLETE")
    

    team = RoundRobinGroupChat(
        [DSLGenerator_cleaned, MetricsAgent_cleaned,KeywordChecker_cleaned],
        termination_condition = max_msg_termination,
    )

    return team
    

    


    
    