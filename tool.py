from config import mapping_dir, result_dir
from autogen_core.tools import FunctionTool
from pathlib import Path
from typing import Dict, List
import pandas as pd
import os
import json
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Optional
import ast
from datetime import datetime
import pytz

async def get_keyword(field_name: str, customer_name: str = "*") -> list:
    try:
        # 构造 URL 和认证信息
        if customer_name != "*":
            index_pattern = customer_name.lower() + "-*"
        if "keyword" not in field_name:
            field_name = field_name + ".keyword"
        url = f"https://jumphost.hkt-ems.com:443/{index_pattern}/_search"
        auth = ('internship', 'P@ssw0rd')

        # 构造查询 DSL
        body = {
            "size": 0,
            "aggs": {
                "all_values": {
                    "terms": {
                        "field": field_name,
                        "size": 10000
                    }
                }
            }
        }

        # 发起 GET 请求（注意：GET 请求中传 body 要加 json.dumps）
        response = requests.get(
            url,
            auth=auth,
            headers={"Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=30,
            verify=False
        )

        response.raise_for_status()  # 若状态码非 2xx，抛出异常

        data = response.json()
        buckets = data.get("aggregations", {}).get("all_values", {}).get("buckets", [])
        return [bucket["key"] for bucket in buckets]
    except Exception as e:
        return [f"Failed to query OpenDistro: {e}"]
Get_keyword_tool = FunctionTool(get_keyword, description="A tool that gets keyword value of a field. ")

def flatten_es_mapping(mapping: dict) -> dict:
    """Flatten Elasticsearch/OpenSearch mapping into {field_path: type} dict."""
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError:
            mapping = ast.literal_eval(mapping)
 
    def _flatten(properties: dict, parent_key=""):
        items = {}
        for field_name, field_info in properties.items():
            full_key = f"{parent_key}.{field_name}" if parent_key else field_name
            field_type = None
            # Check keyword type first
            if "fields" in field_info and "keyword" in field_info["fields"]:
                field_type = field_info["fields"]["keyword"].get("type")
            if not field_type and "type" in field_info:
                field_type = field_info["type"]
            if field_type:
                items[full_key] = field_type
            # Recursively flatten nested fields
            if "properties" in field_info:
                items.update(_flatten(field_info["properties"], full_key))
        return items
 
    if "mappings" in mapping:
        properties = mapping["mappings"].get("properties", {})
    elif isinstance(mapping, dict) and len(mapping) == 1:
        properties = list(mapping.values())[0].get("mappings", {}).get("properties", {})
    else:
        properties = mapping
 
    return _flatten(properties)
 
# =========================
# Tool: Get Flattened Mapping
# =========================
async def get_flattened_mapping(customer_name: str) -> str:
    """
    Retrieve and flatten mapping for a given customer name.
    Saves output to 'flattened_mapping.json' in current dir.
    Returns flattened mapping as JSON string.
    """
    try:
        filename = "flattened_mapping.json"
        output_path = os.path.join(mapping_dir, filename)
        raw_mapping = os.path.join(mapping_dir, "raw_mapping.json")
        index = f"{customer_name.lower()}-*"
        es_url = "https://jumphost.hkt-ems.com:443"
        auth = ("internship", "P@ssw0rd")
        headers = {"Content-Type": "application/json"}
 
        search_url = f"{es_url}/{index}/_mapping"
        response = requests.get(search_url, auth=auth, headers=headers, data="", verify=False)
 
        if response.status_code == 200:
            full_result = response.json()
            if not full_result:
                return "Mapping returned empty."
            first_index = list(full_result.keys())[0]
            mapping_result = {first_index: full_result[first_index]}
            flattened = flatten_es_mapping(mapping_result)
            with open(raw_mapping, 'w', encoding='utf-8') as f:
                json.dump(mapping_result, f, indent=2, ensure_ascii=False)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(flattened, f, indent=2, ensure_ascii=False)
            return json.dumps(flattened, indent=2, ensure_ascii=False)
        else:
            return f"Request failed with status {response.status_code}: {response.text}"
    except Exception as e:
        return f"Failed to query OpenSearch: {e}"
 
Get_mapping_tool = FunctionTool(
    get_flattened_mapping,
    description="Retrieve a customer's flattened mapping from OpenSearch, save to 'flattened_mapping.json', and return as JSON string."
)
 
# =========================
# CSV + Schema Utilities
# =========================
def load_csv_metadata(csv_path: str) -> pd.DataFrame:
    """Load CSV using multiple encodings and clean 'Field' column."""
    encodings = ['utf-8', 'windows-1252', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        import chardet
        with open(csv_path, 'rb') as f:
            result = chardet.detect(f.read())
        df = pd.read_csv(csv_path, encoding=result['encoding'])
    df = df.dropna(subset=['Field'])
    df = df[df['Field'].astype(str).str.strip() != '']
    df = df.fillna('')
    return df
 
def load_flattened_mapping_file(schema_path: str) -> Dict[str, any]:
    """Load JSON file containing flattened mapping."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
 
def filter_csv_by_mapping(field_metadata: pd.DataFrame, schema_fields: set[str]) -> pd.DataFrame:
    """Return only CSV rows where 'Field' matches mapping fields."""
    if field_metadata is None or field_metadata.empty:
        return pd.DataFrame()
    if not schema_fields:
        return field_metadata
    schema_lower_map = {f.lower(): f for f in schema_fields}
    valid_rows = [
        idx for idx, csv_field in field_metadata['Field'].items()
        if csv_field in schema_fields or csv_field.lower() in schema_lower_map
    ]
    return field_metadata.loc[valid_rows].copy()
 
# =========================
# Tool 2: Get Filtered CSV
# =========================
async def get_filtered_csv() -> str:
    """
    Load CSV './all_fields.csv', load mapping from 'flattened_mapping.json',
    filter CSV fields based on mapping, return filtered records as JSON string.
    """
    csv_path = os.path.join(mapping_dir,"all_field.csv")
    schema_path = os.path.join(mapping_dir,"flattened_mapping.json")
    field_metadata = load_csv_metadata(csv_path)
    schema_fields = set(load_flattened_mapping_file(schema_path).keys())
    filtered_df = filter_csv_by_mapping(field_metadata, schema_fields)
    return filtered_df.to_json(orient='records', force_ascii=False, indent=2)
 
Get_csv_tool = FunctionTool(
    get_filtered_csv,
    description="Loads a CSV from 'all_fields.csv', filters it by fields in 'flattened_mapping.json', and returns JSON string."
)

async def Opendistro_search(
    index_name: str,
    query_str: str,
    filename: str = "result.json"
) -> str:
    try:
        output_path = os.path.join(result_dir, filename)
        
        index_name = f"{index_name.lower()}-*"        
        es_url = "https://jumphost.hkt-ems.com:443"
        auth = HTTPBasicAuth("internship", "P@ssw0rd")
        headers = {"Content-Type": "application/json"}

        search_url = f"{es_url}/{index_name}/_search"
        response = requests.get(
            search_url,
            auth=auth,
            headers=headers,
            data=query_str,
            verify=False
        )

        if response.status_code == 200:
            results = response.json()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            output = (f"Query result saved to {output_path}")
            return output
        else:
            return (f"Request failed with status {response.status_code}: {response.text}")
    except Exception as e:
        return (f"Failed to query OpenSearch via requests: {e}")
Opendistro_search_tool = FunctionTool(Opendistro_search, description="A tool that retrieves and stores JSON data from OpenSearch.")


def get_current_time_utc8():
    tz = pytz.timezone('Asia/Shanghai')  # UTC+8
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
Get_time_tool = FunctionTool(get_current_time_utc8, description="A tool that returns the current time in UTC+8 format (Asia/Shanghai).")

from typing import Tuple
# ---------- ③ write Excel ---------------------------------------------------
def agg_json_to_excel(json_path: str | Path,
                    excel_path: str | Path | None = None) -> Tuple[str, pd.DataFrame]:
    excel_path = excel_path or Path(json_path).with_suffix(".xlsx")
    df = agg_json_to_rows(json_path)
    df.to_excel(excel_path, index=False)
    return str(excel_path), df
# ---------- ① recursion to flatten ------------------------------------------
def _walk_agg(node: Dict, path: Dict, rows: List[Dict]):
    for agg_name, agg_val in node.items():
        if isinstance(agg_val, dict) and "buckets" in agg_val:           # ← bucket
            for bucket in agg_val["buckets"]:
                new_path = path.copy() 
                key_value = bucket.get("key_as_string", bucket.get("key"))
                new_path[f"{agg_name}_key"] = key_value
                # new_path[f"{agg_name}_doc_count"] = bucket.get("doc_count", 0)

                has_sub = False
                # metrics in current bucket
                for k, v in bucket.items():
                    if isinstance(v, dict) and "value" in v:             # ← metric
                        new_path[f"{agg_name}_{k}"] = v["value"]

                # nested buckets
                for k, v in bucket.items():
                    if isinstance(v, dict) and "buckets" in v:
                        has_sub = True
                        _walk_agg({k: v}, new_path, rows)

                if not has_sub:
                    rows.append(new_path)

        elif isinstance(agg_val, dict) and "value" in agg_val:           # ← top-level metric
            new_path = path.copy()
            new_path[f"{agg_name}_value"] = agg_val["value"]
            rows.append(new_path)
# ---------- ② entrance of documents ---------------------------------------------------
def agg_json_to_rows(json_path: str | Path) -> pd.DataFrame:
    """
    读取包含 aggregations 的 JSON 文件 → DataFrame
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict] = []
    _walk_agg(data.get("aggregations", {}), {}, rows)
    return pd.json_normalize(rows)                # 列自动展开                 
JSON_TO_EXCEL_tool = FunctionTool(agg_json_to_excel, description="A tool that retrieves a JSON file, decomposes it, and stores it as EXCEL.The required input is the input filename and the output file name is optional.")





