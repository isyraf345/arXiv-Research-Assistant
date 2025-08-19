"""
RAG Pipeline orchestration - combines retrieval and generation
"""
import logging
from typing import List, Dict, Optional, Tuple
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from datetime import datetime

from .config import settings
from .retrieval import RetrievalSystem, QueryProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline that orchestrates retrieval and generation"""
    
    def __init__(self):
        # Initialize components
        self.retrieval_system = RetrievalSystem()
        self.query_processor = QueryProcessor()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key
        )
        
        logger.info(f"RAGPipeline initialized with model: {settings.llm_model}")
    
    def generate_answer(
        self, 
        query: str, 
        max_sources: int = 5,
        include_citations: bool = True
    ) -> Dict[str, any]:
        """
        Generate answer using RAG pipeline
        
        Args:
            query: User's question
            max_sources: Maximum number of source chunks to use
            include_citations: Whether to include source citations
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Process query
            processed_query = self.query_processor.preprocess_query(query)
            filters = self.query_processor.extract_search_filters(processed_query)
            
            logger.info(f"Processing query: '{query}' -> '{processed_query}'")
            
            # Step 2: Retrieve relevant chunks
            relevant_chunks = self.retrieval_system.semantic_search(
                processed_query,
                k=max_sources,
                categories=filters.get('categories')
            )
            
            if not relevant_chunks:
                return {
                    'answer': "I couldn't find any relevant papers for your question. Try a different query or check if papers have been ingested.",
                    'sources': [],
                    'query': query,
                    'processed_query': processed_query,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 3: Build context from retrieved chunks
            context = self._build_context(relevant_chunks)
            
            # Step 4: Generate answer using LLM
            answer = self._generate_llm_response(processed_query, context, include_citations)
            
            # Step 5: Prepare response
            response = {
                'answer': answer,
                'sources': self._format_sources(relevant_chunks),
                'query': query,
                'processed_query': processed_query,
                'num_sources': len(relevant_chunks),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            logger.info(f"Generated answer in {response['processing_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'sources': [],
                'query': query,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            
            context_part = f"""
Source {i}:
Title: {metadata['title']}
Authors: {metadata['authors']}
Published: {metadata['published'][:10]}
Categories: {metadata['categories']}

Content:
{chunk['text']}

---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_llm_response(
        self, 
        query: str, 
        context: str, 
        include_citations: bool = True
    ) -> str:
        """Generate LLM response using retrieved context"""
        
        system_prompt = """You are an expert research assistant specializing in academic papers. Your task is to answer questions based on the provided research paper excerpts.

Instructions:
1. Provide accurate, comprehensive answers based on the given sources
2. If information is not in the sources, clearly state this
3. Maintain academic tone and precision
4. When citing information, reference the source number (e.g., "According to Source 1...")
5. If multiple sources discuss the same topic, synthesize the information
6. Highlight any conflicting information between sources

Focus on being helpful while maintaining scientific accuracy."""

        if include_citations:
            user_prompt = f"""Based on the following research paper excerpts, please answer this question:

Question: {query}

Research Paper Excerpts:
{context}

Please provide a comprehensive answer with proper citations to the source numbers."""
        else:
            user_prompt = f"""Based on the following research paper excerpts, please answer this question:

Question: {query}

Research Paper Excerpts:
{context}

Please provide a comprehensive answer based on this research."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"I encountered an error generating the response: {str(e)}"
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format source information for display"""
        sources = []
        seen_papers = set()
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            arxiv_id = metadata['arxiv_id']
            
            source = {
                'source_number': i,
                'arxiv_id': arxiv_id,
                'title': metadata['title'],
                'authors': metadata['authors'],
                'published': metadata['published'][:10],
                'categories': metadata['categories'],
                'similarity_score': chunk.get('similarity_score', 0),
                'excerpt': chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'],
                'pdf_url': metadata.get('pdf_url', f"https://arxiv.org/pdf/{arxiv_id}.pdf")
            }
            
            sources.append(source)
            seen_papers.add(arxiv_id)
        
        return sources
    
    def summarize_paper(self, arxiv_id: str) -> Dict[str, str]:
        """
        Generate a comprehensive summary of a specific paper
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            Dictionary with different types of summaries
        """
        try:
            paper_context = self.retrieval_system.get_paper_context(arxiv_id)
            if not paper_context:
                return {'error': f"Paper {arxiv_id} not found in database"}
            
            # Combine all chunks (but limit total length)
            all_text = ""
            for chunk in paper_context['chunks']:
                all_text += chunk['text'] + "\n\n"
                if len(all_text) > 10000:  # Limit context size
                    break
            
            # Generate different types of summaries
            summaries = {}
            
            # Executive summary
            exec_prompt = f"""
            Provide a concise executive summary (2-3 paragraphs) of this research paper:
            
            Title: {paper_context['title']}
            Abstract: {paper_context.get('abstract', '')}
            
            Full content:
            {all_text[:8000]}
            
            Focus on: main contribution, methodology, key findings, and significance.
            """
            
            summaries['executive'] = self.llm([HumanMessage(content=exec_prompt)]).content
            
            # Key contributions
            contrib_prompt = f"""
            List the 3-5 key contributions of this paper:
            
            Title: {paper_context['title']}
            Content: {all_text[:6000]}
            
            Format as numbered list of specific contributions.
            """
            
            summaries['contributions'] = self.llm([HumanMessage(content=contrib_prompt)]).content
            
            return {
                'paper_info': {
                    'title': paper_context['title'],
                    'authors': paper_context['authors'],
                    'published': paper_context['published'][:10],
                    'categories': paper_context['categories']
                },
                'summaries': summaries
            }
            
        except Exception as e:
            logger.error(f"Failed to summarize paper {arxiv_id}: {e}")
            return {'error': f"Failed to generate summary: {str(e)}"}

# Convenience functions
def ask_question(question: str, max_sources: int = 5) -> Dict:
    """Simple interface for asking questions"""
    rag = RAGPipeline()
    return rag.generate_answer(question, max_sources=max_sources)

def get_paper_summary(arxiv_id: str) -> Dict:
    """Simple interface for paper summaries"""
    rag = RAGPipeline()
    return rag.summarize_paper(arxiv_id)

if __name__ == "__main__":
    # Test the RAG pipeline
    test_queries = [
        "What are the latest advances in transformer architectures?",
        "How do large language models handle reasoning tasks?",
        "What are the main challenges in computer vision?"
    ]
    
    rag = RAGPipeline()
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = rag.generate_answer(query, max_sources=3)
        
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Sources: {result['num_sources']}")
        print(f"Time: {result['processing_time']:.2f}s")