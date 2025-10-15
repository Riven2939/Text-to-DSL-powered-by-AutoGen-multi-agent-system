import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict, Optional
import os
import json
from config import rag_dir, get_embedding


# Initialize embedding model
embeddings = get_embedding()

import shutil


def DSLExample_db_path(brand) -> str:
    """
    create different ChromaDB path for different wifi brand.
    """
    if brand.lower() == "huawei":
        return os.path.join(rag_dir, "HUAWEI_Example_ChromaDB")
    elif brand.lower() == "aruba":
        return os.path.join(rag_dir, "ARUBA_Example_ChromaDB")

def embed_dsl_examples(
    brand: str,
    db_path: Optional[str] = None,
    embeddings_functions=embeddings  
) -> Dict[str, str]:
    """
    Load DSL examples for a brand, embed them, and save into ChromaDB.   
    """
    db_path = db_path or DSLExample_db_path(brand)

    try:
        if os.path.exists(db_path) and os.listdir(db_path):
            shutil.rmtree(db_path)
            print(f"Existing vector DB at '{db_path}' removed for fresh embedding.")
        
        if brand.lower() == "huawei":
            RagDB_tempath = os.path.join(rag_dir, "DSL_example_HUAWEI_copy.json")
            RagDB_path = os.path.join(rag_dir, "DSL_example_HUAWEI1.json")
        elif brand.lower() == "aruba":
            RagDB_tempath = os.path.join(rag_dir, "DSL_example_ARUBA_copy.json")
            RagDB_path = os.path.join(rag_dir, "DSL_example_ARUBA1.json")



        with open(RagDB_tempath, "r", encoding="utf-8") as f:
            examples = json.load(f)
        with open(RagDB_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        # create Document list
        documents: List[Document] = []
        for item in examples:
            title = item.get("title", "").strip()
            logic = item.get("logic", "").strip()
            description = item.get("description", "").strip()
            dsl = item.get("dsl", {})
            relevant_fields = item.get("relevant_fields", "")
            note = item.get("note", "")

            # embeded content（only for semantic similarity）

            # page_content = f"Title: {title}\nDescription: {description}"
            page_content = f"Title: {title}\nlogic: {logic}"

            # metadata 
            metadata = {
                "title": title,
                "logic": logic,
                "description": description,
                "dsl": json.dumps(dsl, ensure_ascii=False),
                "relevant_fields": relevant_fields,
                "note": note,
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        # create vectorstore
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings_functions,collection_name=brand,collection_metadata={"hnsw:space": "cosine"})
        vectorstore.add_documents(documents)

        return {"response": f"Indexed {len(documents)} DSL examples successfully.", "action": "none"}

    except Exception as e:
        return {"response": f"Error during embedding: {str(e)}", "action": "error"}
    

from typing import Tuple
def query_dsl_examples(
    query: str,
    brand:str,
    db_path: Optional[str] = None,
    top_k: int = 5,
    embeddings=embeddings,
) -> List[Dict]:
    """
    Search DSL examples in a ChromaDB vector store by semantic similarity.

    - Uses the specified `brand` (e.g., huawei, aruba) to locate the correct DB.
    - Retrieves the top_k most relevant examples via cosine distance.
    - Converts distance → similarity score (0–1, higher means more similar).
    - Each result includes: "title", "logic", "dsl", and "score".
    - Returns results as a JSON-formatted string, sorted by similarity (descending).
    
    """
    brand = brand.lower()
    db_path = db_path or DSLExample_db_path(brand=brand)
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings,collection_name=brand,collection_metadata={"hnsw:space": "cosine"})
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    results: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(query, k=top_k)

    out: List[Dict] = []
    for doc, distance in results:
        # 将 cosine 距离 → 相似度；并裁剪到 [0,1]
        try:
            d = float(distance)
        except Exception:
            d = 1.0  # 兜底：当作完全不相似
        sim = max(0.0, min(1.0, 1.0 - d))

        out.append({
            "title": doc.metadata.get("title", ""),
            "logic": doc.metadata.get("logic", ""),
            "dsl": doc.metadata.get("dsl",""),
            "score": sim  # 0~1，越大越相似
        })

    # 明确按相似度降序
    out.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(out, ensure_ascii=False, indent=2)
    

# print(embed_dsl_examples(brand="huawei"))
# print(embed_dsl_examples(brand="aruba"))