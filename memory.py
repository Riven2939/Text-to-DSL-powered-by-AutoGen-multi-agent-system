from pathlib import Path
from typing import Dict, List
import pandas as pd
import os
import json
import requests
from requests.auth import HTTPBasicAuth
from config import mapping_dir


def get_customer(filename: str = "customer.xlsx", customer_column: str = "customer") -> str:
    """
    从Excel中读取客户列表。

    Args:
        excel_path (str): Excel 文件路径。
        customer_column (str): 存储客户名称的列名，默认是 "customer"。

    Returns:
        list: 唯一的客户名称列表。
    """
    excel_path = os.path.join(mapping_dir, filename)
    try:
        df = pd.read_excel(excel_path)
        if customer_column not in df.columns:
            raise ValueError(f"column '{customer_column}' is not exist. Only: {list(df.columns)}")
        customers = df[customer_column].dropna().unique().tolist()
        formatted_output = "customer options: \n" + "\n".join(f"- {c}" for c in customers)
        return formatted_output
    except Exception as e:
        return (f": {e}")
    

def field_description(
    file_path: str,
    columns: list[str],
    item_sep: str = ", ",
    row_sep: str = "\n",
    encoding: str = "ISO-8859-1",
) -> str:
    """
    读取 CSV，将选定列的非空内容拼成一个纯文本字符串。

    :param file_path: CSV 文件路径
    :param columns:   需要拼接的列名列表
    :param item_sep:  同一行内各列之间的分隔符
    :param row_sep:   行与行之间的分隔符
    :param encoding:  文件编码
    :return:          拼接后的字符串
    """
    # 读取 CSV
    df = pd.read_csv(file_path, encoding=encoding)

    # 清理空行空列
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)

    # 仅保留存在且需要的列
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return ""        # 没有任何合法列，直接返回空串

    df = df[valid_cols]

    # 将每行转成字符串
    row_strings: list[str] = []
    for _, row in df.iterrows():
        # 只收集非空字段
        parts = [
            f"{col}{item_sep.split(':')[0]}{row[col]}"  # 保留 “列名: 值” 形式
            if len(item_sep.split(":")) == 2 else f"{row[col]}"
            for col in valid_cols
            if pd.notna(row[col])
        ]
        if parts:
            row_strings.append(item_sep.join(parts))

    # 拼成整体字符串
    all_fields = row_sep.join(row_strings)
    final = all_fields
    return final