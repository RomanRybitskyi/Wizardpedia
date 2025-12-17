from litellm import completion
from src import config

def format_context(results):
    context_text = ""
    sources_display = ""
    for i, (doc_obj, score) in enumerate(results):
        source_name = doc_obj.metadata.get('source', 'Unknown')
        context_text += f"Source [{i+1}] (File: {source_name}): {doc_obj.page_content}\n\n"
        sources_display += f"**Source [{i+1}]** ({source_name}, Score: {score:.2f})\n> {doc_obj.page_content[:200]}...\n\n"
    return context_text, sources_display

def stream_answer(api_key, query, context_text):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer strictly based on the provided Context. "
                "Cite sources using [1], [2] format."
                f"\n\n--- CONTEXT ---\n{context_text}"
            )
        },
        {"role": "user", "content": query}
    ]
    
    return completion(
        model=config.LLM_MODEL_NAME,
        messages=messages,
        api_key=api_key,
        stream=True
    )