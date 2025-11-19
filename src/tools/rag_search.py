import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain.tools import tool


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "llm-teacher-python-db"
TOP_K = 5


TASK_DESCRIPTION = "Найди в технической документации и книгах по Python отрывки, которые максимально релевантны для ответа на следующий запрос пользователя:"


print("[rag] Loading RAG model and Qdrant client...")
_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device)
_qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
_qdrant_client.get_collection(collection_name=COLLECTION_NAME)
print("[rag] RAG model and Qdrant client loaded.")


def _get_detailed_instruct(task: str, query: str) -> str:
    return f'Instruct: {task}\nQuery: {query}'

@tool
def rag_search_tool(query: str) -> str:
    """makes rag search in Qdrant"""
    full_query = _get_detailed_instruct(TASK_DESCRIPTION, query)
    query_vector = _model.encode(full_query, normalize_embeddings=True)
    
    search_results = _qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K,
        with_payload=True
    )
    
    context_str = ""
    if not search_results:
        return "В базе знаний не найдено релевантной информации."
        
    for i, hit in enumerate(search_results):
        payload = hit.payload
        metadata = payload.get('metadata', {})
        context_str += f"--- Источник #{i+1} (Сходство: {hit.score:.4f}) ---\n"
        context_str += f"Название: {metadata.get('source_name', 'N/A')}\n"
        if metadata.get('url'):
            context_str += f"URL: {metadata.get('url')}\n"
        context_str += "Содержимое:\n"
        context_str += payload.get('text', 'Текст отсутствует').strip()
        context_str += "\n--------------------------------------------------\n\n"
        
    return context_str


if __name__ == '__main__':
    test_query = "что такое list comprehension"
    print(f"[rag] test query: '{test_query}'")
    results = rag_search_tool.invoke(test_query)
    print(results)