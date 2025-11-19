# coder.py

import json
import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.prompts.coder import CODER_PROMPT
from src.tools.code_executor import code_executor_tool

class CoderAgentState(TypedDict):
    task_description: str
    rag_context: str
    messages: Annotated[List[BaseMessage], operator.add]
    max_attempts: int
    current_attempt: int
    __end__: Optional[Dict[str, Any]]

coder_llm = ChatOllama(model="hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M", temperature=0.5)
json_parser = JsonOutputParser()

llm_with_tools = coder_llm.bind_tools([code_executor_tool])
execute_tools_node = ToolNode([code_executor_tool])

def call_coder_node(state: CoderAgentState):
    print(f"[coder] try #{state['current_attempt']}")
    
    prompt = CODER_PROMPT.invoke({
        "task_description": state['task_description'],
        "rag_context": state['rag_context'],
        "messages": state['messages']
    })
    
    response = llm_with_tools.invoke(prompt)
    
    return {
        "messages": [response],
        "current_attempt": state['current_attempt'] + 1,
    }

def final_parse_node(state: CoderAgentState):
    print("[coder] parsing final answer")
    final_message_content = state["messages"][-1].content
    
    json_start_index = final_message_content.find('{')
    if json_start_index == -1:
        raise ValueError("[coder] JSON object start ('{') not found in the final message.")
            
    json_string = final_message_content[json_start_index:]
        
    parsed_output = json.loads(json_string, strict=False)
    
    print(f"[coder] parsed output: {parsed_output}")
    
    if isinstance(parsed_output, str):
        parsed_json = json.loads(parsed_output, strict=False)
    else:
        parsed_json = parsed_output

    if not isinstance(parsed_json, dict):
        raise ValueError("[coder] Final parsed output is not a dictionary.")

    tests_content = parsed_json.get("tests", "")
    tests_string = ""
    if isinstance(tests_content, str):
        tests_string = tests_content
    elif isinstance(tests_content, list):
        lines = []
        for test in tests_content:
            if isinstance(test, dict):
                name = test.get('name', 'Unnamed test')
                desc = test.get('description', 'No description.')
                lines.append(f"# Test: {name}\n# Description: {desc}\n")
        tests_string = "\n".join(lines)

    return {
        "__end__": {
            "code": parsed_json.get("code", ""),
            "tests": tests_string,
            "error_log": []
        }
    }

def should_continue(state: CoderAgentState):
    if state["current_attempt"] >= state["max_attempts"]:
        print("[coder] limit of attempts reached. Finishing.")
        return "end_with_error"
        
    last_message: AIMessage = state["messages"][-1]
    
    if last_message.tool_calls:
        print("[coder] agent decided to use a tool. Executing.")
        return "execute_tools"
    else:
        print("[coder] no tool calls. Successful finish.")
        return "final_parse"

graph = StateGraph(CoderAgentState)

graph.add_node("coder_agent", call_coder_node)
graph.add_node("execute_tools", execute_tools_node)
graph.add_node("final_parse", final_parse_node)
graph.add_node("end_with_error", lambda s: {"__end__": {"code": "", "tests": "", "error_log": ["Max attempts reached."]}})

graph.set_entry_point("coder_agent")

graph.add_conditional_edges(
    "coder_agent",
    should_continue,
    {
        "execute_tools": "execute_tools",
        "final_parse": "final_parse",
        "end_with_error": "end_with_error"
    }
)

graph.add_edge("execute_tools", "coder_agent")
graph.add_edge("final_parse", END)
graph.add_edge("end_with_error", END)

coder_agent = graph.compile()


if __name__ == "__main__":
    print("[coder] testing coder agent (tool-use version)")
    
    task = "пример кода с использованием dataclasses в Python."
    
    initial_state = {
        "task_description": task,
        "rag_context": "Dataclasses are a feature in Python for creating classes primarily used to store data. They automatically generate methods like __init__(), __repr__(), and __eq__(). Use the @dataclass decorator.",
        "messages": [],
        "max_attempts": 5,
        "current_attempt": 0,
    }
    
    final_state_result = coder_agent.invoke(initial_state)
    
    print('\n[coder] final state:\n')
    for key, value in final_state_result.items():
        if key == 'messages':
            for message in value:
                print(f"[coder] {message.type}: {message.content}")
        else:
            print(f"[coder] {key}: {value}")
        
        
    final_output = final_state_result.get("__end__", {})

    print("\n[coder] --- final result ---")
    print(f"[coder] code:\n{final_output.get('code', 'N/A')}")
    if final_output.get('error_log'):
        print(f"[coder] final error:\n{final_output['error_log'][-1]}")
