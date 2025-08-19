"""
arXiv Assistant Gradio App
"""
from typing import List, Dict
import gradio as gr

from src.ingestion import ArxivIngestion
from src.rag import RAGPipeline
from datetime import datetime

# Initialize ingestion and RAG
ingestion = ArxivIngestion()
retrieval = RAGPipeline()

# Store ingested papers globally
INGESTED_PAPERS: List[Dict] = []

# -----------------------
# Helper functions
# -----------------------

def ingest_papers(query: str, max_results: int, categories: str):
    category_list = [c.strip() for c in categories.split(",")] if categories else None
    papers = ingestion.search_papers(query=query, max_results=max_results, categories=category_list)
    processed_papers = ingestion.process_papers(papers)
    
    # Update global storage
    global INGESTED_PAPERS
    INGESTED_PAPERS.extend(processed_papers)
    
    status_msg = f"✅ Ingestion Complete!\nPapers Found: {len(papers)} | Successfully Processed: {len(processed_papers)} | Failed: {len(papers)-len(processed_papers)}"
    
    # Richer display
    titles = "\n\n".join([
        f"🔗 [{p['title']}]({p.get('pdf_url', '')})\n"
        f"👤 {', '.join(p.get('authors', []))}\n"
        f"📅 {(p.get('published').strftime('%Y-%m-%d') if isinstance(p.get('published'), datetime) else p.get('published',''))}\n"
        f"📂 {', '.join(p.get('categories', []))}"
        for p in processed_papers
    ]) or "No papers processed."

    # Simple stats
    stats = f"Total Papers Ingested: {len(INGESTED_PAPERS)}\nLast Ingested: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return status_msg, titles, titles, stats  # status, current ingested titles, history display, stats


def ask_question(question: str):
    if not INGESTED_PAPERS:
        return "No papers ingested yet!", "No sources available."
    
    result = retrieval.generate_answer(question)
    answer = result['answer']
    sources = result['sources']
    
    if not sources:
        return answer, "No sources found."
    
    # Richer sources display
    sources_info = []
    for s in sources:
        title = s.get("title", "Unknown Title")
        authors = ", ".join(s.get("authors", [])) if s.get("authors") else "Unknown Authors"
        link = s.get("pdf_url", "")
        score = f"{s.get('similarity_score', 0):.4f}" if "similarity_score" in s else "N/A"
        published = s.get("published", "")[:10]
        categories = ", ".join(s.get("categories", [])) if s.get("categories") else "N/A"
       
        if link:
            sources_info.append(
                f"🔗 [{title}]({link})\n👤 {authors}\n📅 {published}\n📂 {categories}\n📊 Similarity: {score}"
            )
        else:
            sources_info.append(
                f"{title}\n👤 {authors}\n📅 {published}\n📂 {categories}\n📊 Similarity: {score}"
            )
    
    sources_str = "\n\n---\n\n".join(sources_info)
    return answer, sources_str


def get_history_and_stats():
    if not INGESTED_PAPERS:
        return "No papers ingested yet.", "No stats available."
    
    history_list = []
    for p in INGESTED_PAPERS:
        published_str = p.get('published', '')
        if isinstance(published_str, datetime):
            published_str = published_str.strftime('%Y-%m-%d')
        else:
            published_str = str(published_str)[:10]
        
        output = (
            f"🔗 [{p['title']}]({p.get('pdf_url', '')})\n"
            f"👤 {', '.join(p.get('authors', []))}\n"
            f"📅 {published_str}\n"
            f"📂 {', '.join(p.get('categories', []))}"
        )
        history_list.append(output)
    
    history = "\n\n".join(history_list)
    stats = f"Total Papers Ingested: {len(INGESTED_PAPERS)}\nLast Ingested: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return history, stats


# -----------------------
# Gradio UI
# -----------------------

with gr.Blocks() as app:
    
    gr.Markdown("## 📄 arXiv Assistant")
    
    with gr.Tab("Ingest Papers"):
        query_input = gr.Textbox(label="Search Query", placeholder="e.g., Convolutional Neural Networks")
        max_results_input = gr.Number(label="Max Results", value=5)
        category_input = gr.Textbox(label="Categories", placeholder="e.g., cs.AI, cs.LG")
        ingest_btn = gr.Button("Ingest Papers")
        ingest_status = gr.Textbox(label="Status", interactive=False)
        ingested_titles = gr.Textbox(label="Ingested Papers", interactive=False, lines=12)
        
        ingest_btn.click(
            ingest_papers,
            inputs=[query_input, max_results_input, category_input],
            outputs=[ingest_status, ingested_titles, ingested_titles, ingest_status]  # last two for history & stats
        )
    
    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Ask a question about ingested papers")
        ask_btn = gr.Button("Get Answer")
        answer_box = gr.Textbox(label="Answer", interactive=False, lines=8)
        sources_box = gr.Textbox(label="Sources", interactive=False, lines=12)
        
        ask_btn.click(
            ask_question,
            inputs=[question_input],
            outputs=[answer_box, sources_box]
        )
    
    with gr.Tab("History & Stats"):
        history_display = gr.Textbox(label="Ingested Papers", interactive=False, lines=15)
        stats_display = gr.Textbox(label="Statistics", interactive=False)
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(
            get_history_and_stats,
            inputs=[],
            outputs=[history_display, stats_display]
        )

# -----------------------
# Launch
# -----------------------
app.launch(server_name="0.0.0.0", server_port=7860)
