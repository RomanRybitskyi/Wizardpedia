import gradio as gr
from src.engine import HybridRetriever
from src.llm_client import format_context, stream_answer

retriever = HybridRetriever()

def rag_chat_interface(api_key, query, search_mode):
    if not api_key:
        yield "Error: Please enter your Groq API Key.", ""
        return
    if not retriever.documents:
        yield "Error: No documents loaded.", ""
        return
    
    yield "Searching & Reranking...", ""
    
    results = retriever.search(query, mode=search_mode, top_k=3)
    
    context_text, sources_display = format_context(results)
    yield "Generating answer...", sources_display

    try:
        response_stream = stream_answer(api_key, query, context_text)
        partial_answer = ""
        for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                partial_answer += content
                yield partial_answer, sources_display
                
    except Exception as e:
        yield f"API Error: {str(e)}", sources_display

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG System")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(label="Groq API Key", type="password")
            mode_input = gr.Radio(["Hybrid", "Keyword (BM25)", "Semantic (Vectors)"], label="Mode", value="Hybrid")
        with gr.Column(scale=3):
            query_input = gr.Textbox(label="Your Question")
            submit_btn = gr.Button("Submit", variant="primary")
    
    with gr.Row():
        answer_output = gr.Markdown(label="Answer")
        sources_output = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=rag_chat_interface,
        inputs=[api_key_input, query_input, mode_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()