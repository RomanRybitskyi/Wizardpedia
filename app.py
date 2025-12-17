import gradio as gr
from src.engine import HybridRetriever
from src.llm_client import format_context, stream_answer

try:
    retriever = HybridRetriever()
    STATUS_MSG = "App is ready. Documents are uploaded."
except Exception as e:
    retriever = type('obj', (object,), {'documents': []})()
    STATUS_MSG = f"Error of initializing: {e}"

def rag_chat_interface(api_key, query, search_mode):
    if not api_key:
        yield "Error: Please enter your Groq API Key.", ""
        return
    if not retriever.documents:
        yield "Error: No documents loaded. Check server logs.", ""
        return
    
    yield "Searching & Reranking...", ""
    
    try:
        results = retriever.search(query, mode=search_mode, top_k=3)
    except Exception as e:
        yield f"Search Error: {e}", ""
        return
    
    context_text, sources_display = format_context(results)
    if not context_text:
         yield "🤔 I couldn't find any relevant information in the documents.", ""
         return

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

with gr.Blocks(theme=gr.themes.Soft(), title="Advanced RAG System") as demo:
    gr.Markdown("# RAG System")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=200):
            IMAGE_URL = "https://www.google.com/url?sa=t&source=web&rct=j&url=https%3A%2F%2Fwww.bloomsbury.com%2Fuk%2Fdiscover%2Fharry-potter%2Ffun-facts%2F20-fun-facts-about-harry-potter%2F&ved=0CBUQjRxqFwoTCPDf8Pe4xZEDFQAAAAAdAAAAABAI&opi=89978449"
            gr.Image(
                value=IMAGE_URL,
                show_label=False,
                interact=False,
                # height=200, 
                container=False 
            )
        with gr.Column(scale=3):
            gr.Markdown("""
            ### About the service
            This intelligent assistant is designed to work accurately with your knowledge base. It uses the advanced **RAG (Retrieval-Augmented Generation)** approach.
            **How it works:**
            1.  **Hybrid search:** The system searches for documents simultaneously by keywords (BM25) and semantic content (Embeddings).
            2.  **Smart Reranking:** The candidates found are checked by a special neural network (Cross-Encoder) to select only the best results.
            3.  **Response generation:** LLM (Llama 3) forms a reasoned response based solely on the context found, with references to sources.
            """)
    
    gr.Markdown("---") 

    gr.Markdown(f"*{STATUS_MSG}*")
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Groq API Key", 
                type="password", 
                placeholder="gsk_...",
                info="Need for work of LLM"
            )
            mode_input = gr.Radio(
                ["Hybrid (RRF)", "Keyword (BM25)", "Semantic (Vectors)"], 
                label="Search strategy", 
                value="Hybrid (RRF)",
                info="Choose the method"
            )
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Your Question", 
                placeholder="Example: What is known about...",
                lines=3
            )
            submit_btn = gr.Button("Receive the answer 🪄", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Assistant response", header_header=True)
        with gr.Column(scale=1):
            sources_output = gr.Markdown(label="Used sources (Context & Scores)")

    submit_btn.click(
        fn=rag_chat_interface,
        inputs=[api_key_input, query_input, mode_input],
        outputs=[answer_output, sources_output]
    )
    query_input.submit(
        fn=rag_chat_interface,
        inputs=[api_key_input, query_input, mode_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()