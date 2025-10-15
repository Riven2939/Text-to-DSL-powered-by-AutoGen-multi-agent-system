import yaml
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from langchain_openai import AzureOpenAIEmbeddings
import os

#===================================================================
# GPT-4o config
#===================================================================
def get_model_client(config_path="model_config.yaml", temperature: float = 0.0):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["azure"]

    return AzureOpenAIChatCompletionClient(
        azure_deployment=cfg["deployment"],
        model=cfg["model"],
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        temperature=temperature,    
        model_info=cfg["model_info"]
    )

#===================================================================
#GPT-4o-mini    
#===================================================================
def get_mini_model_client(config_path="model_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["mini"]

    return AzureOpenAIChatCompletionClient(
        azure_deployment=cfg["deployment"],
        model=cfg["model"],
        api_version=cfg["api_version"],
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        model_info=cfg["model_info"]
    )

#===================================================================
# Azure embedding config
#===================================================================
def get_embedding(config_path="model_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["azure_embedding"]

    return AzureOpenAIEmbeddings(
        azure_endpoint=cfg["endpoint"],
        api_version=cfg["api_version"],
        api_key=cfg["api_key"],
        azure_deployment=cfg["deployment"]  
    )

#====================================================================
# Directory configuration
#====================================================================

# Get the root directory (the folder this file lives in)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
team_state_dir = os.path.join(BASE_DIR, "team_state")
rag_dir = os.path.join(BASE_DIR, "RAG")
result_dir = os.path.join(BASE_DIR, "result")
mapping_dir = os.path.join(BASE_DIR, "mapping")

# Ensure folders exist (optional)
for folder in [team_state_dir, rag_dir, result_dir, mapping_dir]:
    os.makedirs(folder, exist_ok=True)

print(rag_dir)