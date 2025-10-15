import pandas as pd
import json
import pandas as pd
from config import mapping_dir
import os

def classify_brand(customer_name):
    """
    To classify the wifi brand of current customer
    """
    df = pd.read_excel(os.path.join(mapping_dir, "wifi_customer_brand.xlsx"))
    
    df["Customer"] = df["Customer"].str.lower()
    brand_lookup = dict(zip(df["Customer"], df["WifiBrand"]))
    print(f"brand_lookup: {brand_lookup}")
    print(f"customer_name: {customer_name}")    
    brand = brand_lookup.get(customer_name.lower())
    print(brand)
    return brand


def filter_field_description(brand: str, input_csv: str = "all_field.csv", output_csv: str = "filter_field_description.csv"):
    """
    based on current customer's wifi brand, to filter "all_field.csv".
    """
    input_csv = os.path.join(mapping_dir, input_csv)
    output_csv = os.path.join(mapping_dir, output_csv)    
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '') 
    df["FieldSetID"] = pd.to_numeric(df["FieldSetID"], errors="coerce")

    if brand == "HUAWEI":
        df = df[df["FieldSetID"].isin([0, 1])]
    elif brand == "ARUBA":
        df = df[df["FieldSetID"].isin([0, 2])]

    df.to_csv(output_csv, index=False)

def make_combined_json(output_file=None, **kwargs):
    """
    it takes any number of named inputs (**kwargs) and merges them into one JSON-ready dictionary
    """
    combined = {}
    for key, value in kwargs.items():
        combined[key] = value  
    return combined

import ast

def flatten_es_mapping(mapping: dict) -> dict:
    """
    Flatten an Elasticsearch mapping.
    Prefers `.keyword` when present, recurses through nested `properties`.
    See "flattened_mapping.json". 
    """
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

            # keyword first
            if "fields" in field_info and "keyword" in field_info["fields"]:
                field_type = field_info["fields"]["keyword"].get("type")

            # otherwise, type 
            if not field_type and "type" in field_info:
                field_type = field_info["type"]

            if field_type:
                items[full_key] = field_type

            # if nested, recursion
            if "properties" in field_info:
                items.update(_flatten(field_info["properties"], full_key))

        return items

    # 
    if "mappings" in mapping:
        properties = mapping["mappings"].get("properties", {})
    elif isinstance(mapping, dict) and len(mapping) == 1:
        properties = list(mapping.values())[0].get("mappings", {}).get("properties", {})
    else:
        properties = mapping

    return json.dumps(_flatten(properties), indent=2, ensure_ascii=False)


import re
import json


def extract_json_string(text: str, file_path: str = "DSLQuery.json")-> dict:
    """
    Extract the JSON code block (```json ... ```) from a text string 
    and return it as a Python dict. 
    """
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        raise ValueError("Not found ```json``` ")
    content = match.group(1).strip()
    try:
        return json.loads(content)  
    except json.JSONDecodeError:
        return content
    # save JSON 
    # path = Path(file_path)
    # with path.open("w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)


import time

def stream_data(data: str,time_sleep: float = 0.005):
    """
    Yield text word by word with a short delay, 
    simulating a streaming/typing effect.
    """
    for word in data.split(" "):
        yield word + " "
        time.sleep(time_sleep)


from typing import Union
from pathlib import Path

def write_export_script(
    customer_name: str,
    excel_path: str,
    query: Union[str, dict],
    out_path: str = "opensearch_exporter.py",
) -> str:
    """
    Generate a standalone Python script that queries an OpenSearch index 
    with the given aggregation `query`, then flattens the results and 
    exports them to an Excel file.
    """
    if isinstance(query, dict):
        query_json = json.dumps(query, ensure_ascii=False, indent=2)
    else:
        try:
            query_json = json.dumps(json.loads(query), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            raise ValueError("`query` must be dict or valid JSON_str。")

    # gen script
    script = f'''# -*- coding: utf-8 -*-
"""
Auto-generated OpenSearch aggregation exporter.

Usage:
    python {{__file__}}                          
    python {{__file__}} --customer HKJC --excel out.xlsx --query-file q.json
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import argparse
import sys

def site_agg_to_excel(
    query_str: Union[str, Dict],
    customer_name: str,
    excel_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, pd.DataFrame]:
    index_name = f"{{customer_name.lower()}}-*"
    es_url = "https://jumphost.hkt-ems.com:443"
    auth = HTTPBasicAuth("internship", "P@ssw0rd")  
    headers = {{"Content-Type": "application/json"}}
    search_url = f"{{es_url}}/{{index_name}}/_search"

    payload = query_str if isinstance(query_str, str) else json.dumps(query_str, ensure_ascii=False)

    try:
        resp = requests.get(
            search_url,
            auth=auth,
            headers=headers,
            data=payload,
            verify=False,   
            timeout=60,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to query OpenSearch: {{e}}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"Request failed: {{resp.status_code}} {{resp.text}}")

    try:
        result = resp.json()
    except Exception as e:
        raise ValueError(f"Response is not valid JSON: {{e}}") from e

    def _walk_agg(node: Dict, path: Dict, rows: List[Dict]):
        for agg_name, agg_val in (node or {{}}).items():
            if isinstance(agg_val, dict) and "buckets" in agg_val:
                for bucket in agg_val["buckets"]:
                    new_path = path.copy()
                    key_value = bucket.get("key_as_string", bucket.get("key"))
                    new_path[f"{{agg_name}}_key"] = key_value

                    has_sub = False
                    for k, v in bucket.items():
                        if isinstance(v, dict) and "value" in v:
                            new_path[f"{{agg_name}}_{{k}}"] = v["value"]

                    for k, v in bucket.items():
                        if isinstance(v, dict) and "buckets" in v:
                            has_sub = True
                            _walk_agg({{k: v}}, new_path, rows)

                    if not has_sub:
                        rows.append(new_path)

            elif isinstance(agg_val, dict) and "value" in agg_val:
                new_path = path.copy()
                new_path[f"{{agg_name}}_value"] = agg_val["value"]
                rows.append(new_path)

    rows: List[Dict] = []
    _walk_agg(result.get("aggregations", {{}}), {{}}, rows)
    df = pd.json_normalize(rows) if rows else pd.DataFrame()

    excel_path = Path(excel_path or "result.xlsx")
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False)

    return str(excel_path.resolve()), df


# ---------- default parameter (pass value when generated the py script) ----------
DEFAULT_CUSTOMER = {json.dumps(customer_name, ensure_ascii=False)}
DEFAULT_EXCEL = {json.dumps(excel_path, ensure_ascii=False)}
DEFAULT_QUERY_JSON = r"""{query_json}"""

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run OpenSearch agg and export to Excel.")
    parser.add_argument("--customer", type=str, default=DEFAULT_CUSTOMER, help="customer_name (default: embedded)")
    parser.add_argument("--excel", type=str, default=DEFAULT_EXCEL, help="excel output path (default: embedded)")
    parser.add_argument("--query-file", type=str, default=None, help="path to JSON file as query body (optional)")
    args = parser.parse_args(argv)

    # query-file first, otherwise DEFAULT_QUERY_JSON
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            query_obj = json.load(f)
    else:
        query_obj = json.loads(DEFAULT_QUERY_JSON)

    excel_file, _df = site_agg_to_excel(query_obj, args.customer, args.excel)
    print(f"Exported to: {{excel_file}}")

if __name__ == "__main__":
    sys.exit(main())
'''
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")
    return str(out_path.resolve())