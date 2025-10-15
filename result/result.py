# -*- coding: utf-8 -*-
"""
Auto-generated OpenSearch aggregation exporter.

Usage:
    python {__file__}                          
    python {__file__} --customer HKJC --excel out.xlsx --query-file q.json
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
    index_name = f"{customer_name.lower()}-*"
    es_url = "https://jumphost.hkt-ems.com:443"
    auth = HTTPBasicAuth("internship", "P@ssw0rd")  
    headers = {"Content-Type": "application/json"}
    search_url = f"{es_url}/{index_name}/_search"

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
        raise RuntimeError(f"Failed to query OpenSearch: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"Request failed: {resp.status_code} {resp.text}")

    try:
        result = resp.json()
    except Exception as e:
        raise ValueError(f"Response is not valid JSON: {e}") from e

    def _walk_agg(node: Dict, path: Dict, rows: List[Dict]):
        for agg_name, agg_val in (node or {}).items():
            if isinstance(agg_val, dict) and "buckets" in agg_val:
                for bucket in agg_val["buckets"]:
                    new_path = path.copy()
                    key_value = bucket.get("key_as_string", bucket.get("key"))
                    new_path[f"{agg_name}_key"] = key_value

                    has_sub = False
                    for k, v in bucket.items():
                        if isinstance(v, dict) and "value" in v:
                            new_path[f"{agg_name}_{k}"] = v["value"]

                    for k, v in bucket.items():
                        if isinstance(v, dict) and "buckets" in v:
                            has_sub = True
                            _walk_agg({k: v}, new_path, rows)

                    if not has_sub:
                        rows.append(new_path)

            elif isinstance(agg_val, dict) and "value" in agg_val:
                new_path = path.copy()
                new_path[f"{agg_name}_value"] = agg_val["value"]
                rows.append(new_path)

    rows: List[Dict] = []
    _walk_agg(result.get("aggregations", {}), {}, rows)
    df = pd.json_normalize(rows) if rows else pd.DataFrame()

    excel_path = Path(excel_path or "result.xlsx")
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False)

    return str(excel_path.resolve()), df


# ---------- default parameter (pass value when generated the py script) ----------
DEFAULT_CUSTOMER = "BOCHK"
DEFAULT_EXCEL = "result.xlsx"
DEFAULT_QUERY_JSON = r"""{
  "size": 0,
  "query": {
    "bool": {
      "filter": [
        {
          "range": {
            "@timestamp": {
              "gte": "now-7d/d",
              "lte": "now/d",
              "time_zone": "+08:00",
              "format": "yyyy-MM-dd'T'HH:mm:ss"
            }
          }
        }
      ]
    }
  },
  "aggs": {
    "by_access_point": {
      "terms": {
        "field": "availabilityName.keyword",
        "size": 50
      },
      "aggs": {
        "by_ssid": {
          "terms": {
            "field": "data.SSID.ssidName.keyword",
            "size": 50
          },
          "aggs": {
            "daily_buckets": {
              "date_histogram": {
                "field": "@timestamp",
                "calendar_interval": "day",
                "time_zone": "+08:00"
              },
              "aggs": {
                "max_active_devices_per_day": {
                  "max": {
                    "field": "data.stats.onlineUsers"
                  }
                }
              }
            },
            "max_active_devices": {
              "max_bucket": {
                "buckets_path": "daily_buckets>max_active_devices_per_day"
              }
            }
          }
        }
      }
    }
  }
}"""

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
    print(f"Exported to: {excel_file}")

if __name__ == "__main__":
    sys.exit(main())
