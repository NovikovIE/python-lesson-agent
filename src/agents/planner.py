import json
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.prompts.planner import PLANNER_PROMPT

planner_llm = ChatOllama(model="hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M", temperature=0.2, format="json")

try:
    import json_repair
    def parse_json(text: str):
        return json_repair.loads(text)
except ImportError:
    def parse_json(text: str):
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    
def robust_json_parser(ai_message):
    text = ai_message if isinstance(ai_message, str) else ai_message.content
    
    try:
        return parse_json(text)
    except Exception as e:
        print(f"[planner] !!!WARNING!!! JSON Parsing failed: {e}")
        print(f"[planner] Raw output: {text}...")
        return {
            "sections": [
                {
                    "rag_query": "Error parsing plan",
                    "code_queries": [],
                    "image_query": None
                }
            ]
        }


planner_agent = (
    {"topic": RunnablePassthrough()}
    | PLANNER_PROMPT
    | planner_llm
    | StrOutputParser()
    | robust_json_parser
)

if __name__ == "__main__":
    print("[planner] testing planner agent")
    
    test_topic = "Создай урок про декораторы в Python для новичков"
    print(f"[planner] query: {test_topic}\n")
    
    response = planner_agent.invoke(test_topic)
    
    print("[planner] plan:")
    print('[planner]', json.dumps(response, indent=2, ensure_ascii=False))
    print("\n['planner'] end of testing")
