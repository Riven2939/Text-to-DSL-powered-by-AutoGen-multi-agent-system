from config import get_model_client, mapping_dir
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallSummaryMessage, ToolCallRequestEvent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, MessageFilterAgent, MessageFilterConfig, PerSourceFilter
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

# from autogen_core.model_cont`ext
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
import re

from memory import get_customer, field_description
import os
import json
from function import make_combined_json, filter_field_description

async def get_HistoryMatchTeam(rag_result):
    field_expl = field_description(os.path.join(mapping_dir, "filter_field_description.csv"), ["Field","Explanation","Calculation Method and Data Extraction Method"])


    # MatchEvaluatorAgent_mem
    MatchEvaluatorAgent_mem = ListMemory()
    Matchmem = make_combined_json(field_explanation = field_expl, rag_result = rag_result)
    await MatchEvaluatorAgent_mem.add(MemoryContent(content=Matchmem, mime_type=MemoryMimeType.JSON))

    #AdviseAgent_mem
    Advisor_mem = ListMemory()
    await Advisor_mem.add(MemoryContent(content=field_expl, mime_type=MemoryMimeType.TEXT))
    
    DSLModifier_mem = ListMemory() 
    DSLGenmem = make_combined_json(field_explanation = field_expl)
    await DSLModifier_mem.add(MemoryContent(content=DSLGenmem, mime_type=MemoryMimeType.JSON))

    MatchEvaluatorAgent= AssistantAgent(
        "MatchEvaluatorAgent",
        description="",
        model_client=get_model_client(temperature=0.0),
        tools = [],
        memory= [MatchEvaluatorAgent_mem], 
        reflect_on_tool_use= True,
        # tool_call_summary_format=
        system_message="""
        Your sole objective is to compare the User Requirement against each RAG-retrieved candidate, and rigorously assess whether its **logic** is a **highly similar match** to the User Requirement.
        
        You will be provided {memory}:
            - RAG-retrieved candidate(each with title, logic, and DSL)  
        
        Your Goal:
            1.Read the "logic" section of RAG-retrieved candidate 
            2.Then compare with the user requirement. 
            select the single best-matching candidate **as long as its logic is mostly complete and only requires minor modifications** to satisfy the User Requirement.
        

        ***IMPORTANT***
        If **no candidate** meets the bar of being **highly similar** (i.e. requires moderate or major changes to logic), then output:
            My Chain-of-Thought:... \n
            "No such historical report". 
        
        else:
            Output(If candidate exists):
            ```json
                < title > 
                < logic > 
                < DSL > 
            ```
         """
    )


    Advisor = AssistantAgent(
        "Advisor",
        description="Gathers user requirements, predicts possible report columns",
        model_client = get_model_client(temperature=0.0),
        tools = [],
        memory= [Advisor_mem], 
        reflect_on_tool_use= True,
        # tool_call_summary_format=
        system_message="""
        You are the Advisor.
        Objective
        Given:
            •the User Requirement, and
            •the selected candidate produced by MatchEvaluatorAgent (its title, DSL logic, and DSL query),
            •Field_explanation in your {memory} as reference

        provide precise modification advice to make the DSL fully satisfy the User Requirement. Do not output a full DSL. Produce only targeted, actionable changes with rationale.

        Output format: 
        My advice to modify DSL:
        ```json
        ```
         """
    )

    Advisor_cleaned = MessageFilterAgent(
        name="Advisor",
        wrapped_agent=Advisor,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="MatchEvaluatorAgent", position="last", count=1),
                    PerSourceFilter(source="user",position="first",count=1)
        ]),
    )

    DSLModifier = AssistantAgent(
        "DSLModifier",
        description = "Generates a fully executable OpenSearch DSL query by refining DSL Writer’s draft and ensuring flawless JSON syntax.",
        model_client = get_model_client(temperature=0.0),
        tools = [],
        memory= [DSLModifier_mem], 
        system_message = """
        You are **DSLModifier** USE **Advisor's suggestion** and **field explaination in {memory}** as reference, to do the **minimal modifiction** of the dsl.
        
        Ensure all aggregation keys with clear, meaningful names instead of generic ones like agg1 or bucket2.

        Always do not filter by the field "customer" or "customer_name". 
        set the format of @timestamp as: yyyy-MM-dd'T'HH:mm:ss

        Your Output:
        < DSL in Excutable JSON format >

    """
    )
    DSLModifier_cleaned = MessageFilterAgent(
        name="DSLModifier",
        wrapped_agent=DSLModifier,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="user",position="first",count=1),
                    PerSourceFilter(source="MatchEvaluatorAgent", position="last", count=1),
                    PerSourceFilter(source="Advisor",position="last",count=1)
                    
        ]),
    )

    

    text_termination = TextMentionTermination("No such historical report")
    max_msg_termination = MaxMessageTermination(max_messages=4)
    combined_termination = text_termination | max_msg_termination

    team = RoundRobinGroupChat(
        [MatchEvaluatorAgent,Advisor_cleaned,DSLModifier_cleaned],
        termination_condition= combined_termination,
    )

    return team
    
    
    
    



