import json
import os
import uuid
from typing import List, Dict, Any, Optional, Type, Union
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import tiktoken
import chromadb
from chromadb.config import Settings

load_dotenv()

app = FastAPI(title="Advanced Log Analysis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")


class FilterOperator(str, Enum):
    AND = "AND"
    OR = "OR"
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_EQUALS = "not_equals"
    NOT_CONTAINS = "not_contains"

class FieldFilter(BaseModel):
    key: str
    operator: FilterOperator
    value: Union[str, int, float, List[str]]

class LogicalFilter(BaseModel):
    operator: FilterOperator
    filters: List[Union['LogicalFilter', FieldFilter]]

LogicalFilter.model_rebuild()

class FilterExtractionResponse(BaseModel):
    filters: LogicalFilter
    reasoning: str
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0

class LogAnalysisResult(BaseModel):
    relevant_logs: List[Dict[str, Any]]
    analysis: str
    cost_info: Dict[str, Any]
    filtered_count: int
    total_count: int

class IndexingProgress(BaseModel):
    status: str
    progress: float
    indexed_count: int
    total_count: int
    message: str
    session_id: str

class LogDataFrameProcessor:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.log_uuid_mapping: Dict[str, Dict[str, Any]] = {}
    
    def process_logs_to_dataframe(self, logs: List[Dict[str, Any]]) -> pd.DataFrame:
        flattened_logs = []
        
        for log in logs:
            log_uuid = str(uuid.uuid4())
            flattened_log = self._flatten_json(log, parent_key='', sep='.')
            flattened_log['uuid'] = log_uuid
            self.log_uuid_mapping[log_uuid] = log
            flattened_logs.append(flattened_log)
        
        self.df = pd.DataFrame(flattened_logs)
        self.df = self.df.fillna('')
        return self.df
    
    def _flatten_json(self, nested_json: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for key, value in nested_json.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_json(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                items.append((new_key, str(value)))
            else:
                items.append((new_key, value))
        return dict(items)
    
    def get_dataframe_schema(self) -> List[str]:
        if self.df is None:
            return []
        return self.df.columns.tolist()
    
    def apply_structured_filters(self, filters: LogicalFilter) -> List[str]:
        if self.df is None:
            return []
        mask = self._apply_filter_node(filters)
        filtered_df = self.df[mask]
        return filtered_df['uuid'].tolist()
    
    def _apply_filter_node(self, filter_node: Union[LogicalFilter, FieldFilter]) -> pd.Series:
        if isinstance(filter_node, FieldFilter):
            return self._apply_field_filter(filter_node)
        elif isinstance(filter_node, LogicalFilter):
            return self._apply_logical_filter(filter_node)
        else:
            return pd.Series([True] * len(self.df))
    
    def _apply_field_filter(self, field_filter: FieldFilter) -> pd.Series:
        key = field_filter.key
        operator = field_filter.operator
        value = field_filter.value
        
        if key not in self.df.columns:
            return pd.Series([False] * len(self.df))
        
        column = self.df[key].astype(str)
        
        if operator == FilterOperator.EQUALS:
            return column == str(value)
        elif operator == FilterOperator.CONTAINS:
            return column.str.contains(str(value), case=False, na=False, regex=False)
        elif operator == FilterOperator.STARTS_WITH:
            return column.str.startswith(str(value), na=False)
        elif operator == FilterOperator.ENDS_WITH:
            return column.str.endswith(str(value), na=False)
        elif operator == FilterOperator.NOT_EQUALS:
            return column != str(value)
        elif operator == FilterOperator.NOT_CONTAINS:
            return ~column.str.contains(str(value), case=False, na=False, regex=False)
        elif operator == FilterOperator.IN:
            if isinstance(value, list):
                return column.isin([str(v) for v in value])
            else:
                return column == str(value)
        elif operator == FilterOperator.GREATER_THAN:
            try:
                return pd.to_numeric(column, errors='coerce') > float(value)
            except (ValueError, TypeError):
                return pd.Series([False] * len(self.df))
        elif operator == FilterOperator.LESS_THAN:
            try:
                return pd.to_numeric(column, errors='coerce') < float(value)
            except (ValueError, TypeError):
                return pd.Series([False] * len(self.df))
        elif operator == FilterOperator.GREATER_EQUAL:
            try:
                return pd.to_numeric(column, errors='coerce') >= float(value)
            except (ValueError, TypeError):
                return pd.Series([False] * len(self.df))
        elif operator == FilterOperator.LESS_EQUAL:
            try:
                return pd.to_numeric(column, errors='coerce') <= float(value)
            except (ValueError, TypeError):
                return pd.Series([False] * len(self.df))
        else:
            return pd.Series([False] * len(self.df))
    
    def _apply_logical_filter(self, logical_filter: LogicalFilter) -> pd.Series:
        if not logical_filter.filters:
            return pd.Series([True] * len(self.df))
        
        child_masks = [self._apply_filter_node(child_filter) for child_filter in logical_filter.filters]
        
        if logical_filter.operator == FilterOperator.AND:
            result = child_masks[0]
            for mask in child_masks[1:]:
                result = result & mask
            return result
        elif logical_filter.operator == FilterOperator.OR:
            result = child_masks[0]
            for mask in child_masks[1:]:
                result = result | mask
            return result
        else:
            return child_masks[0] if child_masks else pd.Series([True] * len(self.df))

class VectorSearchService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.indexing_progress = {}
        self.collection = self.client.get_or_create_collection(
            name="log_entries",
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_logs(self, logs: List[Dict[str, Any]], uuids: List[str], batch_size: int = 1000, session_id: str = None) -> None:
        try:
            self.collection.delete()
        except:
            pass
        
        self.collection = self.client.get_or_create_collection(
            name="log_entries",
            metadata={"hnsw:space": "cosine"}
        )
        
        total_logs = len(logs)
        
        if session_id:
            self.indexing_progress[session_id] = {
                "status": "indexing",
                "progress": 0.0,
                "indexed_count": 0,
                "total_count": total_logs,
                "message": "Starting indexing..."
            }
        
        for i in range(0, total_logs, batch_size):
            batch_logs = logs[i:i + batch_size]
            batch_uuids = uuids[i:i + batch_size]
            
            batch_documents = []
            batch_ids = []
            
            for log, log_uuid in zip(batch_logs, batch_uuids):
                flattened_text = self._flatten_and_serialize_json(log)
                batch_documents.append(flattened_text)
                batch_ids.append(log_uuid)
            
            try:
                self.collection.add(
                    documents=batch_documents,
                    ids=batch_ids
                )
                
                indexed_count = min(i + batch_size, total_logs)
                progress = indexed_count / total_logs
                
                if session_id:
                    self.indexing_progress[session_id] = {
                        "status": "indexing",
                        "progress": progress,
                        "indexed_count": indexed_count,
                        "total_count": total_logs,
                        "message": f"Indexed {indexed_count}/{total_logs} logs..."
                    }
                
                print(f"Indexed batch {i//batch_size + 1}/{(total_logs + batch_size - 1)//batch_size} ({len(batch_documents)} logs)")
            except Exception as e:
                print(f"Error indexing batch {i//batch_size + 1}: {e}")
                if session_id:
                    self.indexing_progress[session_id] = {
                        "status": "error",
                        "progress": 0.0,
                        "indexed_count": 0,
                        "total_count": total_logs,
                        "message": f"Error during indexing: {str(e)}"
                    }
                continue
        
        if session_id:
            self.indexing_progress[session_id] = {
                "status": "completed",
                "progress": 1.0,
                "indexed_count": total_logs,
                "total_count": total_logs,
                "message": "Indexing completed!"
            }
    
    def get_indexing_progress(self, session_id: str) -> Dict[str, Any]:
        return self.indexing_progress.get(session_id, {
            "status": "not_found",
            "progress": 0.0,
            "indexed_count": 0,
            "total_count": 0,
            "message": "Session not found"
        })
    
    def semantic_search(self, query: str, n_results: int = 20) -> List[str]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['ids'][0] if results['ids'] else []
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _flatten_and_serialize_json(self, log: Dict[str, Any]) -> str:
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, str(v) if v is not None else ''))
            return dict(items)
        
        flattened = flatten_dict(log)
        sorted_items = sorted(flattened.items())
        
        text_parts = []
        for key, value in sorted_items:
            if value and value.strip():
                text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)

def _retry_with_backoff(func, max_retries=3):
    import time
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)

client = openai.OpenAI()
MODEL_NAME = "gpt-4.1-mini"

def generate_structured(
    messages: list[dict],
    response_format: None | Type[BaseModel] = None,
    model: str = MODEL_NAME,
    return_usage: bool = False,
):
    def _generate_structured():
        completion = client.responses.parse(
            model=model,
            input=messages,
            text_format=response_format,
        )
        if return_usage:
            return completion.output_parsed, completion.usage
        return completion.output_parsed

    return _retry_with_backoff(_generate_structured)

class FinalAnalysisResponse(BaseModel):
    relevant_log_indices: List[int]
    analysis: str


class StructuredLLMService:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
        self.df_processor = None
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_columns",
                    "description": "Analyze DataFrame columns to get unique values for filter creation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of column names to analyze for unique values"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation of why these columns are relevant for the incident"
                            }
                        },
                        "required": ["columns", "reasoning"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "create_filters",
                    "description": "Create inclusive structured filters based on column analysis. Filters should be broad and capture all potentially relevant logs for downstream analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filters": {
                                "type": "object",
                                "description": "Hierarchical filter structure with operator and filters. Use OR operators liberally to be inclusive. Prefer 'contains' over 'equals' for broader matching. Example: {\"operator\": \"OR\", \"filters\": [{\"key\": \"service.name\", \"operator\": \"contains\", \"value\": \"cart\"}, {\"operator\": \"AND\", \"filters\": [{\"key\": \"severity\", \"operator\": \"in\", \"value\": [\"ERROR\", \"WARN\"]}]}]}"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation of the inclusive filter design and why it captures all potentially relevant logs"
                            }
                        },
                        "required": ["filters"]
                    }
                }
            }
        ]
    
    def extract_structured_filters(self, prompt: str, available_fields: List[str]) -> FilterExtractionResponse:
        if not self.df_processor:
            return self._simple_filter_extraction(prompt, available_fields)
        
      
        system_prompt = f"""You are an expert log analyst. Your task is to analyze an incident description and identify the most relevant DataFrame columns for filtering logs and create filters based on the actual incident values.

Available columns in the log dataset:
{', '.join(available_fields)}

IMPORTANT: The goal is to create INCLUSIVE filters that capture ALL potentially relevant logs for downstream analysis. It's better to include too many logs than to miss important ones.

You need to identify which columns are most relevant for the incident and should be analyzed for their unique values. Focus on columns that would help filter logs related to the incident.

Common important column types:
- Service identifiers: resource_attributes.service.name, resource_attributes.k8s.deployment.name
- Severity levels: fields.severity_text, fields.severity_number  
- Namespaces: resource_attributes.k8s.namespace.name
- Log content: body (we shouldn't use this column for analyze_columns tool, as there will be too many unique values)
- Timestamps: timestamp
- Error indicators in various fields

Process:
1. First, use analyze_columns tool to get unique values for relevant columns
2. Then, use create_filters tool to build inclusive filters based on the actual values found

**'filters' field is a MUST when calling create_filters tool.**

FILTER FORMAT EXAMPLE:
When creating filters, use this exact hierarchical structure:

{{
  "operator": "OR",
  "filters": [
    {{
      "operator": "AND",
      "filters": [
        {{"key": "resource_attributes.service.name", "operator": "contains", "value": "cart"}},
        {{"key": "fields.severity_text", "operator": "in", "value": ["ERROR", "WARN", "FATAL"]}}
      ]
    }},
    {{
      "operator": "AND",
      "filters": [
        {{"key": "resource_attributes.service.name", "operator": "contains", "value": "payment"}},
        {{"key": "body", "operator": "contains", "value": "fail"}}
      ]
    }},
    {{"key": "body", "operator": "contains", "value": "timeout"}}
  ]
}}

Available operators: AND, OR, equals, contains, starts_with, ends_with, greater_than, less_than, in, not_equals, not_contains

INCLUSIVE FILTERING GUIDELINES:
- Use "contains" instead of "equals" for broader matching
- Use "OR" operators liberally to capture more logs
- Include related services (e.g., if incident mentions "cart", also include "order", "payment")
- Include multiple severity levels (not just ERROR)
- Use "in" operator for multiple values: {{"operator": "in", "value": ["ERROR", "WARN"]}}"""

        
        user_message = f"Incident Description: {prompt}\n\nIdentify relevant columns and create inclusive filters."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            return self._multi_step_analysis(messages, prompt)
        except Exception as e:
            print(f"Error in multi-step filter extraction: {e}")
            return self._simple_filter_extraction(prompt, available_fields)
    
    def _multi_step_analysis(self, messages: List[Dict], original_prompt: str) -> FilterExtractionResponse:
        max_steps = 10
        total_input_tokens = 0
        total_output_tokens = 0
        
        for step in range(max_steps):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="required" if step == 0 else "auto",
                    temperature=0.1
                )
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens
                
                if response.choices[0].message.tool_calls:
                    tool_results = []
                    
                    for tool_call in response.choices[0].message.tool_calls:
                        if tool_call.function.name == "analyze_columns":
                            result = self._handle_analyze_columns(tool_call)
                            tool_results.append(result)
                        elif tool_call.function.name == "create_filters":
                            result = self._handle_create_filters(tool_call)
                            if result.get("final_result"):
                                # Add token usage to the final result
                                final_result = result["final_result"]
                                final_result.input_tokens = total_input_tokens
                                final_result.output_tokens = total_output_tokens
                                final_result.total_tokens = total_input_tokens + total_output_tokens
                                return final_result
                            tool_results.append(result)
                    
                    messages.append({
                        "role": "assistant", 
                        "content": response.choices[0].message.content,
                        "tool_calls": response.choices[0].message.tool_calls
                    })
                    
                    for tool_result in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result["tool_call_id"],
                            "content": tool_result["content"]
                        })
                
                else:
                    break
                    
            except Exception as e:
                print(f"Error in step {step}: {e}")
                break
        
        fallback_result = self._create_fallback_filter(original_prompt)
        fallback_result.input_tokens = total_input_tokens
        fallback_result.output_tokens = total_output_tokens
        fallback_result.total_tokens = total_input_tokens + total_output_tokens
        return fallback_result
    
    def _handle_analyze_columns(self, tool_call) -> Dict:
        try:
            args = json.loads(tool_call.function.arguments)
            columns = args.get("columns", [])
            reasoning = args.get("reasoning", "")
            
            column_values = {}
            for column in columns:
                if column in self.df_processor.df.columns:
                    unique_vals = self.df_processor.df[column].dropna().astype(str).unique().tolist()
                    column_values[column] = unique_vals
                else:
                    column_values[column] = []
            
            result_content = json.dumps({
                "column_values": column_values,
                "analysis": f"Retrieved unique values for {len(columns)} columns: {', '.join(columns)}",
                "reasoning": reasoning,
                "next_step": "Now use create_filters tool to build inclusive filters based on these actual values."
            }, indent=2)
            
            return {
                "tool_call_id": tool_call.id,
                "content": result_content
            }
            
        except Exception as e:
            return {
                "tool_call_id": tool_call.id,
                "content": json.dumps({"error": f"Failed to analyze columns: {str(e)}"})
            }
    
    def _handle_create_filters(self, tool_call) -> Dict:
        try:
            args = json.loads(tool_call.function.arguments)
            filters_dict = args.get("filters", {})
            reasoning = args.get("reasoning", "")
            
            filters = self._dict_to_logical_filter(filters_dict)
            
            result = FilterExtractionResponse(
                filters=filters,
                reasoning=reasoning
            )
            
            return {
                "tool_call_id": tool_call.id,
                "content": "Filters created successfully",
                "final_result": result
            }
            
        except Exception as e:
            return {
                "tool_call_id": tool_call.id,
                "content": json.dumps({"error": f"Failed to create filters: {str(e)}"})
            }
    
    def _dict_to_logical_filter(self, filter_dict: Dict) -> LogicalFilter:
        if "operator" not in filter_dict:
            raise ValueError("Filter must have an operator")
        
        operator = FilterOperator(filter_dict["operator"])
        filters = []
        
        for filter_item in filter_dict.get("filters", []):
            if "key" in filter_item:
                field_filter = FieldFilter(
                    key=filter_item["key"],
                    operator=FilterOperator(filter_item["operator"]),
                    value=filter_item["value"]
                )
                filters.append(field_filter)
            else:
                nested_filter = self._dict_to_logical_filter(filter_item)
                filters.append(nested_filter)
        
        return LogicalFilter(operator=operator, filters=filters)
    
    def _simple_filter_extraction(self, prompt: str, available_fields: List[str]) -> FilterExtractionResponse:
        """Fallback simple filter extraction without multi-step analysis"""
        system_prompt = f"""Create inclusive filters for log analysis. Available fields: {', '.join(available_fields)}
        
Use broad, inclusive filters that capture all potentially relevant logs for further analysis downstream."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create inclusive filters for: {prompt}"}
            ]
            
            response = generate_structured(
                messages=messages,
                response_format=FilterExtractionResponse,
                model=MODEL_NAME
            )
            return response
        except:
            return self._create_fallback_filter(prompt)
    
    def _create_fallback_filter(self, prompt: str) -> FilterExtractionResponse:
        """Create a basic fallback filter"""
        empty_filter = LogicalFilter(
            operator=FilterOperator.AND,
            filters=[]
        )
        return FilterExtractionResponse(
            filters=empty_filter,
            reasoning=f"Fallback filter for prompt: {prompt}"
        )

class HybridLogAnalysisService:
    def __init__(self):
        self.df_processor = LogDataFrameProcessor()
        self.vector_service = VectorSearchService()
        self.structured_llm = StructuredLLMService()
        self.structured_llm.df_processor = self.df_processor
        self.encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
    
    def analyze_logs(self, logs: List[Dict[str, Any]], prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if not logs:
            return {
                "relevant_logs": [],
                "analysis": "No logs provided for analysis.",
                "cost_info": self._calculate_cost(0, 0),
                "filtered_count": 0,
                "total_count": 0
            }
        
        df = self.df_processor.process_logs_to_dataframe(logs)
        uuids = df['uuid'].tolist()
        
        should_index = True
        if session_id:
            progress = self.vector_service.get_indexing_progress(session_id)
            if progress.get("status") == "completed":
                should_index = False
                print(f"✅ Reusing pre-indexed data for session {session_id}")
        
        if should_index:
            print("🔄 Indexing logs (no pre-indexing found)")
            self.vector_service.index_logs(logs, uuids)
        else:
            print("⚡ Skipping indexing - using pre-indexed data")
        
        available_fields = self.df_processor.get_dataframe_schema()
        filter_extraction = self.structured_llm.extract_structured_filters(prompt, available_fields)
        
        structured_uuids = self.df_processor.apply_structured_filters(filter_extraction.filters)
        semantic_uuids = self.vector_service.semantic_search(prompt, n_results=200)
        combined_uuids = self._combine_search_results(structured_uuids, semantic_uuids)
        
        relevant_logs = [
            self.df_processor.log_uuid_mapping[uuid_val] 
            for uuid_val in combined_uuids 
            if uuid_val in self.df_processor.log_uuid_mapping
        ]
        
        analysis_result = self._perform_final_analysis(relevant_logs, prompt)
        
        # Combine token usage from filter extraction and final analysis
        filter_input_tokens = filter_extraction.input_tokens or 0
        filter_output_tokens = filter_extraction.output_tokens or 0
        
        analysis_input_tokens = analysis_result["cost_info"]["input_tokens"]
        analysis_output_tokens = analysis_result["cost_info"]["output_tokens"]
        
        total_input_tokens = filter_input_tokens + analysis_input_tokens
        total_output_tokens = filter_output_tokens + analysis_output_tokens
        
        combined_cost_info = self._calculate_cost(total_input_tokens, total_output_tokens)
        
        return {
            "relevant_logs": analysis_result["relevant_logs"],
            "analysis": analysis_result["analysis"],
            "cost_info": combined_cost_info,
            "filtered_count": len(relevant_logs),
            "total_count": len(logs)
        }
    
    def _combine_search_results(self, structured_uuids: List[str], semantic_uuids: List[str]) -> List[str]:
        combined = structured_uuids.copy()
        for uuid_val in semantic_uuids:
            if uuid_val not in combined:
                combined.append(uuid_val)
        return combined[:1000]
    
    def _perform_final_analysis(self, logs: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        if not logs:
            return {
                "relevant_logs": [],
                "analysis": "No relevant logs found after filtering.",
                "cost_info": self._calculate_cost(0, 0)
            }
        
        analysis_logs = logs[:200]
        
        system_prompt = """You are an expert log analyst. Analyze the logs in context of the incident.

Your task:
1. Identify the most relevant logs for understanding the incident (return their indices)
2. Provide comprehensive markdown analysis including:
   - What the logs reveal about the incident
   - Key patterns, errors, or anomalies found
   - Potential root causes based on evidence
   - Recommended next steps for investigation

Format your analysis in markdown with clear sections and bullet points for readability.

Respond in the following format:

"relevant_log_indices": [list of indices of most relevant logs],
"analysis": "comprehensive markdown analysis of the incident with sections for findings, root causes, and next steps"
"""
        
        user_prompt = f"""
Incident Report: {prompt}

Complete logs to analyze ({len(analysis_logs)} entries):
{json.dumps(analysis_logs, indent=2)}

Please provide a comprehensive analysis and identify the most relevant logs by their indices.
"""
        
        # Count input tokens
        input_text = system_prompt + user_prompt
        input_tokens = len(self.encoding.encode(input_text))
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            structured_response, usage = generate_structured(
                messages=messages,
                response_format=FinalAnalysisResponse,
                model=MODEL_NAME,
                return_usage=True
            )
            
            # Use actual token usage from the API response
            final_input_tokens = usage.input_tokens if hasattr(usage, 'input_tokens') else input_tokens
            final_output_tokens = usage.output_tokens if hasattr(usage, 'output_tokens') else len(self.encoding.encode(structured_response.analysis))
            cost_info = self._calculate_cost(final_input_tokens, final_output_tokens)
            
            relevant_indices = structured_response.relevant_log_indices
            relevant_logs = [
                analysis_logs[i] for i in relevant_indices 
                if 0 <= i < len(analysis_logs)
            ]
            
            return {
                "relevant_logs": relevant_logs,
                "analysis": structured_response.analysis,
                "cost_info": cost_info
            }
            
        except Exception as e:
            return {
                "relevant_logs": analysis_logs[:5],
                "analysis": f"## Analysis Error\n\nAnalysis completed with errors: {str(e)}\n\nShowing first 5 logs as fallback.",
                "cost_info": self._calculate_cost(final_input_tokens if 'final_input_tokens' in locals() else input_tokens, 0)
            }
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Calculate GPT-4.1-mini API cost"""
        input_cost_per_1k = 0.0008
        output_cost_per_1k = 0.0032
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }

analysis_service = HybridLogAnalysisService()

@app.post("/analyze", response_model=LogAnalysisResult)
async def analyze_logs(
    file: UploadFile = File(...), 
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    try:
        content = await file.read()
        lines = content.decode('utf-8').strip().split('\n')
        
        all_logs = []
        for line in lines:
            if line.strip():
                try:
                    log_entry = json.loads(line)
                    all_logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
        
        if not all_logs:
            raise HTTPException(status_code=400, detail="No valid logs found in file")
        
        result = analysis_service.analyze_logs(all_logs, prompt, session_id)
        return LogAnalysisResult(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing logs: {str(e)}")

@app.post("/index-logs", response_model=IndexingProgress)
async def index_logs_endpoint(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        content = await file.read()
        lines = content.decode('utf-8').strip().split('\n')
        
        all_logs = []
        for line in lines:
            if line.strip():
                try:
                    log_entry = json.loads(line)
                    all_logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
        
        if not all_logs:
            raise HTTPException(status_code=400, detail="No valid logs found in file")
        
        df_processor = LogDataFrameProcessor()
        df = df_processor.process_logs_to_dataframe(all_logs)
        uuids = df['uuid'].tolist()
        
        import threading
        def index_in_background():
            analysis_service.vector_service.index_logs(all_logs, uuids, batch_size=500, session_id=session_id)
        
        thread = threading.Thread(target=index_in_background)
        thread.start()
        
        return IndexingProgress(
            status="indexing",
            progress=0.0,
            indexed_count=0,
            total_count=len(all_logs),
            message="Indexing started...",
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting indexing: {str(e)}")

@app.get("/index-progress/{session_id}", response_model=IndexingProgress)
async def get_indexing_progress(session_id: str):
    progress_data = analysis_service.vector_service.get_indexing_progress(session_id)
    return IndexingProgress(
        status=progress_data["status"],
        progress=progress_data["progress"],
        indexed_count=progress_data["indexed_count"],
        total_count=progress_data["total_count"],
        message=progress_data["message"],
        session_id=session_id
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)