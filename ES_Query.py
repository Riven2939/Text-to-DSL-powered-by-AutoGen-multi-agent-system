import json
import pandas as pd
import os
from typing import Dict, List, Any, Set
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import chardet



class QueryFields(BaseModel):
    filtering_fields: List[str] = Field(description="Fields used for filtering/WHERE conditions")
    grouping_fields: List[str] = Field(description="Fields used for grouping/aggregation")
    metric_fields: List[str] = Field(description="Fields used for calculations/metrics")



class EnhancedElasticsearchQueryGenerator:
    def __init__(self, azure_openai_endpoint: str, azure_openai_key: str, 
                 deployment_name: str, api_version: str = "2024-02-01"):
        """Initialize with Azure OpenAI credentials"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_openai_endpoint,
            azure_deployment=deployment_name,
            openai_api_key=azure_openai_key,
            openai_api_version=api_version,
            temperature=0
        )
        
        # Load your field metadata CSV
        self.field_metadata = None
        self.database_schema = None
        self.schema_fields = set()  # Store actual schema field names
        
    def load_field_metadata(self, csv_path: str):
        """Load the CSV containing field names, explanations, and calculation methods with encoding handling"""
        try:
            # First try UTF-8
            self.field_metadata = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Try Windows-1252 encoding (most common for Excel CSV files)
                self.field_metadata = pd.read_csv(csv_path, encoding='windows-1252')
            except UnicodeDecodeError:
                try:
                    # Try Latin-1 as fallback
                    self.field_metadata = pd.read_csv(csv_path, encoding='latin-1')
                except UnicodeDecodeError:
                    # Auto-detect encoding as last resort
                    with open(csv_path, 'rb') as f:
                        result = chardet.detect(f.read())
                        detected_encoding = result['encoding']
                    
                    self.field_metadata = pd.read_csv(csv_path, encoding=detected_encoding)
        
        # Clean the data to handle NaN values
        # Remove rows where 'Field' column is NaN or empty
        self.field_metadata = self.field_metadata.dropna(subset=['Field'])
        
        # Remove rows where Field is empty string or whitespace only
        self.field_metadata = self.field_metadata[self.field_metadata['Field'].astype(str).str.strip() != '']
        
        # Fill NaN values in other columns with empty strings
        self.field_metadata = self.field_metadata.fillna('')
        
    def load_database_schema(self, schema_path: str):
        """Load database schema information and extract valid field names"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.database_schema = json.load(f)
            
            # Extract field names from schema
            self.schema_fields = self._extract_schema_fields(self.database_schema)
            
        except FileNotFoundError:
            self.database_schema = None
            self.schema_fields = set()
        except json.JSONDecodeError:
            self.database_schema = None
            self.schema_fields = set()
    
    def _extract_schema_fields(self, schema: Dict[str, Any]) -> Set[str]:
        """Extract all field names from Elasticsearch mapping schema"""
        fields = set()
        
        def extract_fields_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "properties" and isinstance(value, dict):
                        # This is a properties section - extract field names
                        for field_name in value.keys():
                            full_path = f"{path}.{field_name}" if path else field_name
                            fields.add(full_path)
                            # Recursively extract nested fields
                            extract_fields_recursive(value[field_name], full_path)
                    elif key == "mappings" and isinstance(value, dict):
                        # Handle mappings section
                        extract_fields_recursive(value, path)
                    elif isinstance(value, dict):
                        # Continue recursion for nested objects
                        extract_fields_recursive(value, path)
        
        extract_fields_recursive(schema)
        return fields
    
    def _filter_fields_by_schema(self) -> pd.DataFrame:
        """Filter CSV fields to only include those that exist in the schema"""
        if self.field_metadata is None:
            return None
            
        if not self.schema_fields:
            return self.field_metadata
        
        # Check which CSV fields exist in the schema
        csv_fields = self.field_metadata['Field'].astype(str).str.strip()
        
        # Create case-insensitive mapping for schema fields
        schema_fields_lower = {field.lower(): field for field in self.schema_fields}
        
        valid_rows = []
        
        for idx, csv_field in csv_fields.items():
            csv_field_lower = csv_field.lower()
        
            # ONLY Direct match and case-insensitive match - NO PARTIAL MATCHING
            if csv_field in self.schema_fields:
                valid_rows.append(idx)
            # Case-insensitive match
            elif csv_field_lower in schema_fields_lower:
                valid_rows.append(idx)
        
        # Return only rows with valid schema fields
        filtered_metadata = self.field_metadata.loc[valid_rows].copy()
        
        return filtered_metadata
    
    def extract_query_fields(self, natural_language_query: str) -> QueryFields:
        """Step 1: Extract filtering, grouping, and metric fields from natural language"""
        
        # Use only fields that exist in the schema
        valid_field_metadata = self._filter_fields_by_schema()
        
        # Create field context from valid fields only
        field_context = ""
        if valid_field_metadata is not None and len(valid_field_metadata) > 0:
            field_context_lines = []
            for _, row in valid_field_metadata.iterrows():
                field_name = str(row['Field']) if pd.notna(row['Field']) else "Unknown"
                explanation = str(row['Explanation']) if pd.notna(row['Explanation']) else "No explanation"
                method = str(row['Calculation Method and Data Extraction Method']) if pd.notna(row['Calculation Method and Data Extraction Method']) else "No method"
                field_context_lines.append(f"- {field_name}: {explanation} (Method: {method})")
            field_context = "\n".join(field_context_lines)
        else:
            field_context = "No valid fields found in schema"
        
        parser = PydanticOutputParser(pydantic_object=QueryFields)
        
        # Add schema fields to the prompt for better validation
        schema_field_list = ", ".join(sorted(self.schema_fields)) if self.schema_fields else "No schema provided"
        
        prompt_template = PromptTemplate(
            template="""You are an expert at analyzing natural language queries and extracting relevant database fields.


IMPORTANT: You must ONLY use field names from the "Available Fields" list below. Do not use any field names not explicitly listed.


Available Fields and Their Meanings:
{field_context}

Natural Language Query: {query}


Based on the query, identify:
1. FILTERING FIELDS: Fields that should be used to filter/restrict the data (WHERE conditions)
2. GROUPING FIELDS: Fields that should be used to group/aggregate the data (GROUP BY)
3. METRIC FIELDS: Fields that should be calculated/measured (SUM, COUNT, AVG, etc.)


CRITICAL: Only use field names that exist in the "Available Fields" list above. If a concept in the query cannot be mapped to an available field, leave that category empty rather than guessing field names.


{format_instructions}
""",
            input_variables=["field_context", "query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        prompt = prompt_template.format(
            field_context=field_context,
            #schema=json.dumps(self.database_schema, indent=2) if self.database_schema else "No schema provided",
            query=natural_language_query
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        print("##########################################################")
        print(f"Extract_field: {response.usage_metadata}")
        print("##########################################################")
        
        return parser.parse(response.content)
    
    def validate_and_correct_fields(self, query_fields: QueryFields) -> QueryFields:
        """Validate extracted fields against schema and CSV metadata"""
        if self.field_metadata is None:
            return query_fields
        
        # Get valid fields that exist in both CSV and schema
        valid_field_metadata = self._filter_fields_by_schema()
        if valid_field_metadata is None or len(valid_field_metadata) == 0:
            return QueryFields(filtering_fields=[], grouping_fields=[], metric_fields=[])
        
        # Create set of valid field names (case-insensitive)
        valid_fields_lower = {str(field).lower(): str(field) 
                             for field in valid_field_metadata['Field'] 
                             if pd.notna(field)}
        
        def validate_field(field_name: str) -> str:
            """Validate field exists in schema - EXACT MATCH ONLY"""
            if not isinstance(field_name, str):
                field_name = str(field_name)
        
            field_lower = field_name.lower().strip()
    
            # ONLY Direct match - NO PARTIAL MATCHING
            if field_lower in valid_fields_lower:
                return valid_fields_lower[field_lower]
    
            # NO PARTIAL MATCHING - remove field if not exact match
            return None
        
        # Validate all field categories
        validated_filtering = [f for f in [validate_field(field) for field in query_fields.filtering_fields] if f is not None]
        validated_grouping = [f for f in [validate_field(field) for field in query_fields.grouping_fields] if f is not None]
        validated_metrics = [f for f in [validate_field(field) for field in query_fields.metric_fields] if f is not None]
        
        return QueryFields(
            filtering_fields=validated_filtering,
            grouping_fields=validated_grouping,
            metric_fields=validated_metrics
        )
    
    def generate_elasticsearch_query(self, query_fields: QueryFields, 
                                   natural_language_query: str) -> Dict[str, Any]:
        """Step 2: Generate Elasticsearch DSL from validated fields"""
        
        # Double-check that all fields exist in schema
        all_query_fields = (query_fields.filtering_fields + 
                           query_fields.grouping_fields + 
                           query_fields.metric_fields)
        
        if not all_query_fields:
            return {
                "error": "No valid fields found for query generation",
                "message": "All requested fields were filtered out because they don't exist in the schema"
            }
        
        prompt_template = PromptTemplate(
            template="""You are an expert at generating Elasticsearch Query DSL.


Original Query: {query}
Filtering Fields: {filtering_fields}
Grouping Fields: {grouping_fields}  
Metric Fields: {metric_fields}


Available Field Metadata:
{field_context}


IMPORTANT: All field names provided above have been validated to exist in the Elasticsearch schema.


Generate a complete Elasticsearch Query DSL that:
1. size: 0 (no hits, only aggregations)
2. Uses the filtering fields for query/filter contexts
3. Uses the grouping fields for aggregations (use "terms" aggregation for grouping, fields being aggregation should be in keyword type, do not set size)
4. Uses the metric fields for metric aggregations (sum, avg, count, max, serial_diff etc.) (Follow the calculation methods provided in the field metadata, do not make your own assumptions)
5. Include necessary aggregations


Special rules: 
If metrics aggregation uses **serial_diff**, ensure that the last grouping aggregation before metrics aggregation is by a short time interval; if serial_diff is not used, this requirement does not apply. Do not do any grouping without being told.



Return ONLY valid JSON for the Elasticsearch query DSL. Do not include any explanation or markdown formatting.
""",
            input_variables=["query", "filtering_fields", "grouping_fields", 
                           "metric_fields", "field_context"]
        )
        
        # Use only validated fields for field context
        valid_field_metadata = self._filter_fields_by_schema()
        field_context = ""
        if valid_field_metadata is not None:
            field_context_lines = []
            for _, row in valid_field_metadata.iterrows():
                field_name = str(row['Field']) if pd.notna(row['Field']) else "Unknown"
                explanation = str(row['Explanation']) if pd.notna(row['Explanation']) else "No explanation"
                method = str(row['Calculation Method and Data Extraction Method']) if pd.notna(row['Calculation Method and Data Extraction Method']) else "No method"
                field_context_lines.append(f"- {field_name}: {explanation} (Method: {method})")
            field_context = "\n".join(field_context_lines)
        
        prompt = prompt_template.format(
            query=natural_language_query,
            filtering_fields=", ".join(query_fields.filtering_fields),
            grouping_fields=", ".join(query_fields.grouping_fields),
            metric_fields=", ".join(query_fields.metric_fields),
            field_context=field_context
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        print(f"LLM Response: {response.usage_metadata}")

        try:
            # Clean up the response content
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {"error": "Failed to parse Elasticsearch query", "raw_response": response.content}
    
    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """Enhanced pipeline with schema validation and minimal output"""
        
        # Step 1: Extract field categories (already filtered by schema)
        raw_fields = self.extract_query_fields(natural_language_query)
        
        # Step 2: Validate and correct fields against schema
        validated_fields = self.validate_and_correct_fields(raw_fields)
        
        # Output only the chosen fields
        print("Chosen Fields:")
        print(f"  Filtering Fields: {validated_fields.filtering_fields}")
        print(f"  Grouping Fields: {validated_fields.grouping_fields}")
        print(f"  Metric Fields: {validated_fields.metric_fields}")
        
        # Step 3: Generate Elasticsearch query
        es_query = self.generate_elasticsearch_query(validated_fields, natural_language_query)
        
        return {
            "raw_fields": raw_fields.model_dump(),
            "validated_fields": validated_fields.model_dump(),
            "elasticsearch_query": es_query,
            "schema_fields_count": len(self.schema_fields),
            "valid_csv_fields_count": len(self._filter_fields_by_schema()) if self._filter_fields_by_schema() is not None else 0
        }


def generate_elasticsearch_query_from_natural_language(query: str, mapping_path:str, description_path:str, save_folder: str = ".") -> str:
    """
    Convert natural language query to Elasticsearch DSL query and save to JSON file.
    
    Args:
        query (str): Natural language query string
        save_folder (str): Folder path to save the JSON file, default is current directory
        
    Returns:
        str: the Elasticsearch query
    """
    
    # Hardcoded configuration
    AZURE_OPENAI_ENDPOINT = "https://bnk9.openai.azure.com/"
    AZURE_OPENAI_KEY = "ce8db1ddf4c548eeb72507885186bf4f"
    DEPLOYMENT_NAME = "gpt-4o"

    CSV_FILE_PATH = os.path.join(save_folder,description_path)
    SCHEMA_FILE_PATH = os.path.join(save_folder, mapping_path)
    
    try:
        # Initialize the generator
        generator = EnhancedElasticsearchQueryGenerator(
            azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_openai_key=AZURE_OPENAI_KEY,
            deployment_name=DEPLOYMENT_NAME
        )
        
        # Load the required files
        generator.load_field_metadata(CSV_FILE_PATH)
        generator.load_database_schema(SCHEMA_FILE_PATH)
        
        # Process the query
        result = generator.process_query(query)
        
        # Get the Elasticsearch query
        elasticsearch_query = result["elasticsearch_query"]
        
    except Exception as e:
        # Create error JSON if something goes wrong
        elasticsearch_query = {
            "error": "Failed to generate Elasticsearch query",
            "message": str(e),
            "query": query
        }
    
    # Return only the elasticsearch_query
    return elasticsearch_query


# Usage example:
# if __name__ == "__main__":
#     # Example usage
#     user_query = "Throughput of an network in the last 7 days"
#     saved_filename = generate_elasticsearch_query_from_natural_language(user_query)
    
#     print(f"Elasticsearch query saved to: {saved_filename}")
