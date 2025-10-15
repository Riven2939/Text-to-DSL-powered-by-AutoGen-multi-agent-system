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
from config import team_state_dir, get_model_client, mapping_dir
from memory import get_customer, field_description
import os
import json
from function import make_combined_json


async def get_fieldTeam(task = None, customer_name = None,mapping = None,user_require = None, ):
    #memory 
    #RagAgent
    
    field_expl = field_description(os.path.join(mapping_dir, "filter_field_description.csv"), ["Field", "Explanation"])

    #FieldSelector
    FieldSelector1_mem = ListMemory()
    FSmem = make_combined_json(field_explanation = field_expl,mapping = mapping)
    await FieldSelector1_mem.add(MemoryContent(content=FSmem, mime_type=MemoryMimeType.JSON))

    FieldSelector2_mem = ListMemory()
    FSmem = make_combined_json(field_explanation = field_expl)
    await FieldSelector2_mem.add(MemoryContent(content=FSmem, mime_type=MemoryMimeType.JSON))

    FieldCriticalAgent1_mem = ListMemory()
    FC1_mem = make_combined_json(field_explanation = field_expl)
    await FieldCriticalAgent1_mem.add(MemoryContent(content=FC1_mem, mime_type=MemoryMimeType.JSON))

    FieldCriticalAgent2_mem = ListMemory()
    knowledge= "Note: In most index designs, each document comes from one specific device or data source only. For example, if a document contains client-side data, it will not also contain AP-side data. Therefore, fields from different device types usually cannot coexist in the same document. Be careful not to approve field combinations that would never appear together in real data."
    FC2_mem = make_combined_json(knowledge = knowledge, field_explanation = field_expl)
    await FieldCriticalAgent2_mem.add(MemoryContent(content=FC2_mem, mime_type=MemoryMimeType.JSON))

    FieldSelector1 = AssistantAgent(
    "FieldSelector1",
    description="Suggests all fields that can be used in both the filter (query) and aggregation (agg) parts of a DSL query, based on user requirements and index mapping.",
    model_client=get_model_client(temperature=1.0),
    tools = [],
    memory= [FieldSelector1_mem], 
    system_message="""
        You are FieldSelector1, responsible for identifying **ALL** relevant fields that **may appear** in the *query* or *aggregation* sections of a DSL query.

        You will be provided {memory}:
            • field_explanation: 
                field explanations for fields.
            
        Your Task:
            1. Read the user requirement carefully.
            2. READ field_explanation carefully.
            3. Based on the user requirement, identify **ALL the possible field candidates** from field_explanation:

        OUTPUT FORMAT:
        ```json
        {\n
        "Field_Candidates": {\n
            "field name1": "reason",\n
            "field name2": "reason"\n
            ...
            }\n
        }
        ```
        """
    )
    FieldSelector1_cleaned = MessageFilterAgent(
        name="FieldSelector1",
        wrapped_agent=FieldSelector1,
        filter=MessageFilterConfig(per_source = [
                    # PerSourceFilter(source="FieldCriticalAgent", position="last", count=1),
                    PerSourceFilter(source="user", position = "first", count = 1)              
        ]),
    )

    FieldSelector2 = AssistantAgent(
    "FieldSelector2",
    description="Suggests all fields that can be used in both the filter (query) and aggregation (agg) parts of a DSL query, based on user requirements and index mapping.",
    model_client=get_model_client(temperature=0.3),
    tools = [],
    memory= [FieldSelector2_mem], 
    system_message="""
        You are FieldSelector2, responsible for identifying all relevant fields that may appear in the *query* or *aggregation* sections of a DSL query.

        You will be provided {memory}:
            A field description as available field.        

        Your Task:
            1. Read the user requirement carefully.
            2. Check field description.
            3. Based on the user requirement, identify from field_explanation:
                - FILTERING FIELDS: Fields that should be used to filter/restrict the data (WHERE conditions)
                - GROUPING FIELDS: Fields that should be used to group/aggregate the data (GROUP BY)
                - METRIC FIELDS: Fields that should be calculated/measured (SUM, COUNT, AVG, etc.)

        OUTPUT FORMAT:
        ```json
        Field_Candidates: 
        field name: reason
        ```
        """
    )
    FieldSelector2_cleaned = MessageFilterAgent(
        name="FieldSelector2",
        wrapped_agent=FieldSelector2,
        filter=MessageFilterConfig(per_source = [
                    # PerSourceFilter(source="FieldCriticalAgent", position="last", count=1),
                    PerSourceFilter(source="user", position = "first", count = 1)              
        ]),
    )

    FieldCriticalAgent1 = AssistantAgent(
    "FieldCriticalAgent1",
    description="Suggests all fields that can be used in both the filter (query) and aggregation (agg) parts of a DSL query, based on user requirements and index mapping.",
    model_client=get_model_client(temperature=0.2),
    tools = [],
    memory= [FieldCriticalAgent1_mem],
    system_message="""
        You are **FieldCriticalAgent1**—the reviewer that validates the field set proposed by FieldSelectorAgent.

        You will be provided:
            • {memory}:
                A field description as reference.
            • Chat History:
                FieldSelector1 and FieldSelector2's message.

        **analyze FieldSelector's thought and message**, using all available resources. Focus on:
            1. **Fitness**: Are the selected fields truly necessary based on the user requirement and field description? Are there any important fields **missing or wrongly included**? 
            2. **Duplication**: Are there any redundant or duplicated fields in the proposed set?
            3. Other Insights by yourself(optional).

        
        From the candidate fields, select the **minimal complete set that can satisfy the requirement**
        You may add fields from only field_explanation if necessary.

        OUTPUT FORMAT:
        THOUGHT(be consise):
        < Your thought >
        MESSAGE:
        < Your suggestions > 
        """
    )
    FieldCriticalAgent1_cleaned = MessageFilterAgent(
        name="FieldCriticalAgent1",
        wrapped_agent=FieldCriticalAgent1,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="FieldSelector1", position="last", count=1),
                    PerSourceFilter(source="FieldSelector2",position="last", count=1),
                    PerSourceFilter(source="user", position = "first", count = 1)   
        ]),
    )

    FieldCriticalAgent2 = AssistantAgent(
    "FieldCriticalAgent2",
    description="Suggests all fields that can be used in both the filter (query) and aggregation (agg) parts of a DSL query, based on user requirements and index mapping.",
    model_client=get_model_client(temperature=0.2),
    tools = [],
    memory= [FieldCriticalAgent2_mem],
    system_message="""
        You are **FieldCriticalAgent2**—the reviewer that validates the field set proposed by FieldSelectorAgent.
        You will be provided:
            • knowledge and field_explanation as reference in your {memory}:
            • Chat History:
                FieldSelector1 and FieldSelector2's message.
        
        Comment on FieldCriticalAgent's thought and message base on the following criteriors: 
            1. **Conflicts / Mutual-Exclusion** - detect fields that cannot coexist. (Use the field_explanation as reference)
            2. **Conflict-Free Optimal Selection** – From the candidate fields, select a subset that avoids all conflicts and best matches the user requirement. 

        From the candidate fields, select the **minimal complete set that can satisfy the requirement**
        You may add fields from only field_explanation if necessary.
            
        OUTPUT FORMAT:
        THOUGHT(be consise):
        < Your thought >
        MESSAGE:
        < Your suggestions > 
        
        """
    )
    FieldCriticalAgent2_cleaned = MessageFilterAgent(
        name="FieldCriticalAgent2",
        wrapped_agent=FieldCriticalAgent2,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="FieldSelector1", position="last", count=1),
                    PerSourceFilter(source="FieldSelector2",position="last", count=1),
                    PerSourceFilter(source="user", position = "first", count = 1) 
        ]),
    )
    
    FieldFinalizer = AssistantAgent(
    "FieldFinalizer",
    description="Suggests all fields that can be used in both the filter (query) and aggregation (agg) parts of a DSL query, based on user requirements and index mapping.",
    model_client=get_model_client(temperature=0.0),
    tools = [],
    memory= [],
    system_message="""
        You are FieldFinalizer. Your job is to make a well-reasoned final decision on which fields should be included.
        You will be provided:
            • Chat History:
                FieldCriticalAgent1's analysis and recommendations  
                FieldCriticalAgent2's analysis and recommendations
                user's requirement
        Based on the user_requirement:
        1. **Carefully consider both FieldCriticalAgents’ thoughts** 
        2. base on user requirement, identify:
            - FILTERING FIELDS: Fields that should be used to filter/restrict the data (WHERE conditions)
            - GROUPING FIELDS: Fields that should be used to group/aggregate the data (GROUP BY)
            - METRIC FIELDS: Fields that should be calculated/measured (SUM, COUNT, AVG, etc.)

        OUTPUT FORMAT: 
        ```json
            FILTERING FIELDS list:[]
            GROUPING FIELDS list:[]
            METRIC FIELDS list :[] 
        ```
        < DONE >
        """
    )
    FieldFinalizer_cleaned = MessageFilterAgent(
        name="FieldFinalizer",
        wrapped_agent=FieldFinalizer,
        filter=MessageFilterConfig(per_source = [
                    PerSourceFilter(source="FieldCriticalAgent1", position="last", count=1),
                    PerSourceFilter(source="FieldCriticalAgent2", position="last", count=1),
                    PerSourceFilter(source="user", position = "first", count = 1)
        ]),
    )
    
    
    text_termination = TextMentionTermination("DONE")
    max_msg_termination = MaxMessageTermination(max_messages=6)

    team = RoundRobinGroupChat(
        [FieldSelector1_cleaned, FieldSelector2_cleaned, FieldCriticalAgent1_cleaned, FieldCriticalAgent2_cleaned, FieldFinalizer_cleaned],
        termination_condition=max_msg_termination,
    )

    return team




            
    
    
   
            
   