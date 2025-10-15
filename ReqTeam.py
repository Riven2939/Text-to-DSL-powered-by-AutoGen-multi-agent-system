# from fastapi import FastAPI, WebSocket
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

from tool import Get_csv_tool, Get_keyword_tool, Get_mapping_tool, Get_time_tool
from memory import get_customer, field_description
import os
import json
from config import team_state_dir, get_model_client, get_mini_model_client
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent, ToolCallSummaryMessage, ToolCallRequestEvent,MemoryQueryEvent
import aiofiles



async def get_Reqteam(stage,state_path):
    #memory
    #RequirementAgent
    customer_finder_agent_mem = ListMemory()
    customer_name = get_customer()
    await customer_finder_agent_mem.add(MemoryContent(content=customer_name, mime_type=MemoryMimeType.TEXT))
 
    customer_finder_agent = AssistantAgent(
    name="CustomerFinderAgent",
    description="Identifies the customer name from query or asks directly if not present.",
    model_client=get_model_client(temperature=0.1),
    tools=[Get_mapping_tool],
    memory=[customer_finder_agent_mem],
    reflect_on_tool_use=True,
    system_message="""
        You are CustomerFinderAgent.  
        Your job: find "customer_name" in the query. Then retrieve the mapping for that customer with the Get_mapping_tool.
        You MUST call the mapping tool before confirming the customer name.
 
        Steps:
        1. Recalling {memory} to retrieve the available customer name list.
        2. Compare the retrieved names against the customer name(s) mentioned in the user’s query.
        3. If a customer name is found in the query:
        - Call Get_mapping_tool for that customer.
        - Proceed to the output step.
        4. If no customer name is found in the query:
        - Ask the user to specify the customer name in the following format:
            <Question:[your question]?>
        - If the user asks for available options, output the list of names retrieved from Get_customer_tool for them to choose from.
        5. After the customer name is confirmed, you MUST call Get_mapping_tool before proceeding.
 
        When done, output exactly:
        CUSTOMER_FOUND:
        ```json
        <customer_name>
        ```
        """
    )
    
    filter_finder_agent = AssistantAgent(
    name="FilterFinderAgent",
    description="Finds what needs to be filtered, verifies using CSV & keyword tools, interacts with user if unclear.",
    model_client=get_model_client(temperature=0.7),
    tools=[Get_csv_tool, Get_keyword_tool,Get_time_tool],
    memory=[],
    reflect_on_tool_use=True,
    system_message="""
        You are FilterFinderAgent.  
        Goal: Identify and confirm all filters.
        Tool: get_keyword(field_name: str, customer_name: str)
        Use the tool Get_time_tool to retrieve current date and time. 
        Ask Question one at a time. Wait till user answers before proceeding to the next question.
        
 
        Steps:
        1. Always start by calling Get_csv_tool to retrieve the list of available fields for filtering.
        2. Read the query to detect any filtering needs.  
        - Examples: "status is active", "location = Hong Kong", "date after 2024-06-01"  
        Restriction: Do not Include or Guess or ASK for **any field name**, unless metioned by **USER**

        3. If filtering is needed:
        a. If the query specifies an exact value for a filter:
            - Call Get_keyword_tool with that field name to verify the value is available.
            - If the value is invalid, tell user and ask for clarification.
        b. If the query does not provide an exact value OR if the value is unclear:
            - Call Get_keyword_tool with that field name to retrieve all possible options.
            - Present the list of retrieved options to the user for selection.
        4. If the user asks for "options" or "choices" for a specific filter field they mentioned:
        - Identify the field name.
        - Call Get_keyword_tool for that field.
        - Return the option list to the user for selection.
        5. Interact with the user to confirm what they want to filter.  
        Use the format:  Question:[your question]?
        6. If the user says "no" to adding filters, do not assume any default filters.
        
        7. Once all filters are confirmed, output exactly in this format:
        
        FILTERS_CONFIRMED:
        ```json
        <"filter condition" OR "exact field name(if mentioned by user)">
        ```
        <PASS to analyzer>
        """
    )
 
    requirements_analyzer = AssistantAgent(
    name="RequirementsFinalizer",
    description="Finalizes the JSON response after all data is collected.",
    model_client=get_model_client(temperature=0.3),
    tools=[],
    memory=[],
    reflect_on_tool_use=False,
    system_message="""
        You are RequirementsFinalizer.
        You must read the output of the CustomerFinderAgent and FilterFinderAgent and the query.   
        Ask Question one at a time. Wait till user answers before proceeding to the next question.
        Goal: Output the final JSON as:
        Restriction: Do not Include or Guess or ASK for **any field name**, unless metioned by **USER**
    
        ```json
        {
        "customer": "",
        "summary": "Concise summary of what the user wants",
        "mentioned_metrics": [],
        "mentioned_filters": [],
        "mentioned_groupings": [],
        "mentioned_time_range": [],
        "mentioned_methodology": []
        }
        ```    
 
        "metrics", "filters","grouping", "time_range" are required and if missing you must ask for clarification. Say :"<Question:[your question]?>" to ask user anything.
        For the last field if not provided, ask user : "<Question: Do you want to provide a methodology for this request? If no, I will assume standard methodology.>"(Do not output the above json)
        If answer is no, leave it blank.
 
        Before outputing the JSON, you should finish all your question first. Do not output the JSON while asking questions.
        When done, output exactly:
        \n
        ```json
        {
        "customer": "",
        "summary": "Concise summary of what the user wants",
        "mentioned_metrics": [],
        "mentioned_filters": [],
        "mentioned_groupings": [],
        "mentioned_time_range": [],
        "mentioned_methodology": []
        }
        ```
        """
    )
    
 
    if stage == "customer_finder_agent":
        if not os.path.exists(state_path):
            return customer_finder_agent
        async with aiofiles.open(state_path, "r") as file:
            state = json.loads(await file.read())
        await customer_finder_agent.load_state(state)
        return customer_finder_agent
    
    elif stage == "filter_finder_agent":
        if not os.path.exists(state_path):
            return filter_finder_agent
        async with aiofiles.open(state_path, "r") as file:
            state = json.loads(await file.read())
        await filter_finder_agent.load_state(state)
        return filter_finder_agent
    
    elif stage == "requirements_analyzer":
        if not os.path.exists(state_path):
            return requirements_analyzer
        async with aiofiles.open(state_path, "r") as file:
            state = json.loads(await file.read())
        await requirements_analyzer.load_state(state)
        return requirements_analyzer
 
 