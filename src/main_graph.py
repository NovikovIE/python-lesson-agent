from typing import TypedDict, Annotated, List
import json

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src.agents.planner import planner_agent
from src.tools.rag_search import rag_search_tool
from src.prompts.synthesizer import SYNTHESIZER_PROMPT
from src.agents.coder import coder_agent
from src.tools.image_search import image_tool 
from src.utils import clean_llm_output 

class NewLessonState(TypedDict):
    user_topic: str
    plan: List[dict]
    gathered_data: str
    final_lesson: str

synthesizer_llm = ChatOllama(model="hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M", temperature=0.3)


def planner_node(state: NewLessonState):
    print("[main] 1. planning")
    plan_json = planner_agent.invoke(state["user_topic"])
    print(f"[main] Plan created: {plan_json['sections']}")
    return {"plan": plan_json['sections']}

def data_gathering_node(state: NewLessonState):
    print("[main] 2. data gathering, section by section")
    plan = state["plan"]
    all_gathered_results = []

    for i, section in enumerate(plan):
        print(f"\n[main] section {i+1}/{len(plan)}")
        
        section_content = []
        
        rag_query = section.get("rag_query")
        print(f"[main] RAG: {rag_query}...")
        rag_context = rag_search_tool.invoke(rag_query)
        section_content.append(f"### Теория: {rag_query}\n{rag_context}")
        
        image_query = section.get("image_query")
        if image_query:
            print(f"[main] ImgSearch Looking for: {image_query}")
            saved_path = image_tool.search_and_rerank(image_query)
            
            if saved_path:
                all_gathered_results.append(f"![Иллюстрация к теме]({saved_path})")
            else:
                print("[main] no image found")

        for code_query in section["code_queries"]:
            print(f"[main] Code query: {code_query}")
            
            initial_state = {
                "task_description": code_query,
                "rag_context": rag_context,
                "messages": [],
                "max_attempts": 5,
                "current_attempt": 0,
            }
            
            final_state = coder_agent.invoke(initial_state)
            code_output = final_state.get("__end__", {})
            
            if code_output and not code_output.get("error_log"):
                formatted_code = f"#### Код: {code_query}\n\n```python\n{code_output.get('code', '')}\n```"
                if code_output.get('tests'):
                    formatted_code += f"\n\n#### Тесты:\n```python\n{code_output.get('tests', '')}\n```"
                section_content.append(formatted_code)
            else:
                section_content.append(f"#### Код: {code_query}\n\n**ОШИБКА:** Не удалось сгенерировать код.")
        
        all_gathered_results.append("\n\n".join(section_content))

    return {"gathered_data": "\n\n---\n\n".join(all_gathered_results)}

def synthesizer_node(state: NewLessonState):
    print("\n[main] 3. synthesizer writing")
    
    synthesizer_chain = SYNTHESIZER_PROMPT | synthesizer_llm | StrOutputParser() | clean_llm_output
    
    final_lesson_content = synthesizer_chain.invoke({
        "topic": state["user_topic"],
        "gathered_data": state["gathered_data"]
    })
    
    return {"final_lesson": final_lesson_content}

main_workflow = StateGraph(NewLessonState)

main_workflow.add_node("planner", planner_node)
main_workflow.add_node("gather_data", data_gathering_node)
main_workflow.add_node("synthesize_lesson", synthesizer_node)

main_workflow.set_entry_point("planner")
main_workflow.add_edge("planner", "gather_data")
main_workflow.add_edge("gather_data", "synthesize_lesson")
main_workflow.add_edge("synthesize_lesson", END)

lesson_generator_app = main_workflow.compile()

def generate_lesson(topic: str) -> str:
    initial_state = {"user_topic": topic}
    final_state = lesson_generator_app.invoke(initial_state)
    return final_state["final_lesson"]