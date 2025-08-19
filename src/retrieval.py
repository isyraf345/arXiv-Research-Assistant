"""
Retrieval system for finding relevant paper chunks
"""
import logging
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalSystem:
    """Handles semantic search and retrieval of relevant paper chunks"""
    
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            self.collection = self.client.get_collection("arxiv_papers")
        except Exception:
            logger.warning("Collection 'arxiv_papers' not found. Run ingestion first.")
            self.collection = None
        
        # Initialize embedding model (same as ingestion)
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        logger.info("RetrievalSystem initialized")
    
    def semantic_search(
        self, 
        query: str, 
        k: int = None,
        categories: Optional[List[str]] = None,
        date_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Perform semantic search for relevant paper chunks
        
        Args:
            query: User's search query
            k: Number of results to return
            categories: Filter by arXiv categories
            date_filter: Filter by date (e.g., "2024")
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.collection:
            logger.error("No collection available for search")
            return []
        
        k = k or settings.retrieval_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Build where clause for filtering
            where_clause = {}
            if categories:
                # Note: ChromaDB where clauses are limited, so we'll filter post-retrieval
                pass
            
            # Perform vector search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k * 2,  # Get more results for post-filtering
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            chunks = []
            for i in range(len(results['ids'][0])):
                chunk = {
                    'chunk_id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                }
                
                # Apply post-retrieval filters
                if categories:
                    chunk_categories = chunk['metadata']['categories'].split(', ')
                    if not any(cat in chunk_categories for cat in categories):
                        continue
                
                if date_filter:
                    if date_filter not in chunk['metadata']['published']:
                        continue
                
                chunks.append(chunk)
            
            # Sort by similarity and return top k
            chunks = sorted(chunks, key=lambda x: x['similarity_score'], reverse=True)[:k]
            
            logger.info(f"Found {len(chunks)} relevant chunks for query: '{query[:50]}...'")
            return chunks
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_paper_context(self, arxiv_id: str) -> Dict:
        """
        Get all chunks and metadata for a specific paper
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            Paper context with all chunks
        """
        if not self.collection:
            return {}
        
        try:
            results = self.collection.get(
                where={"arxiv_id": arxiv_id},
                include=['documents', 'metadatas']
            )
            
            if not results['ids']:
                return {}
            
            # Organize chunks by index
            chunks = []
            for i in range(len(results['ids'])):
                chunks.append({
                    'chunk_id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            # Sort by chunk index
            chunks = sorted(chunks, key=lambda x: x['metadata']['chunk_index'])
            
            # Return paper context
            first_metadata = chunks[0]['metadata']
            return {
                'arxiv_id': arxiv_id,
                'title': first_metadata['title'],
                'authors': first_metadata['authors'],
                'abstract': first_metadata.get('abstract', ''),
                'categories': first_metadata['categories'],
                'published': first_metadata['published'],
                'chunks': chunks,
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Failed to get paper context for {arxiv_id}: {e}")
            return {}
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Combine semantic and keyword search
        
        Args:
            query: Search query
            k: Number of results
            keyword_weight: Weight for keyword matching
            semantic_weight: Weight for semantic similarity
            
        Returns:
            Ranked list of relevant chunks
        """
        k = k or settings.retrieval_k
        
        # Get semantic results
        semantic_results = self.semantic_search(query, k=k*2)
        
        # Simple keyword scoring
        query_words = set(query.lower().split())
        
        for chunk in semantic_results:
            text_words = set(chunk['text'].lower().split())
            keyword_overlap = len(query_words.intersection(text_words)) / len(query_words)
            
            # Combine scores
            semantic_score = chunk['similarity_score']
            hybrid_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_overlap)
            chunk['hybrid_score'] = hybrid_score
        
        # Re-rank by hybrid score
        hybrid_results = sorted(semantic_results, key=lambda x: x['hybrid_score'], reverse=True)
        
        return hybrid_results[:k]
    
    def find_similar_papers(self, arxiv_id: str, k: int = 5) -> List[Dict]:
        """
        Find papers similar to a given paper
        
        Args:
            arxiv_id: Reference paper ID
            k: Number of similar papers to return
            
        Returns:
            List of similar papers
        """
        # Get the paper's abstract as query
        paper_context = self.get_paper_context(arxiv_id)
        if not paper_context:
            return []
        
        abstract = paper_context.get('abstract', '')
        if not abstract:
            # Use first chunk as fallback
            if paper_context['chunks']:
                abstract = paper_context['chunks'][0]['text'][:500]
        
        # Search for similar content, excluding the same paper
        results = self.semantic_search(abstract, k=k*2)
        
        # Filter out chunks from the same paper and group by paper
        similar_papers = {}
        for chunk in results:
            chunk_arxiv_id = chunk['metadata']['arxiv_id']
            if chunk_arxiv_id == arxiv_id:
                continue
                
            if chunk_arxiv_id not in similar_papers:
                similar_papers[chunk_arxiv_id] = {
                    'arxiv_id': chunk_arxiv_id,
                    'title': chunk['metadata']['title'],
                    'authors': chunk['metadata']['authors'],
                    'categories': chunk['metadata']['categories'],
                    'published': chunk['metadata']['published'],
                    'max_similarity': chunk['similarity_score'],
                    'relevant_chunks': 1
                }
            else:
                similar_papers[chunk_arxiv_id]['relevant_chunks'] += 1
                if chunk['similarity_score'] > similar_papers[chunk_arxiv_id]['max_similarity']:
                    similar_papers[chunk_arxiv_id]['max_similarity'] = chunk['similarity_score']
        
        # Sort by similarity and return top k
        similar_list = list(similar_papers.values())
        similar_list = sorted(similar_list, key=lambda x: x['max_similarity'], reverse=True)
        
        return similar_list[:k]
    
    def get_collection_statistics(self) -> Dict:
        """Get statistics about the vector collection"""
        if not self.collection:
            return {'status': 'No collection available'}
        
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample = self.collection.get(
                limit=min(100, count),
                include=['metadatas']
            )
            
            if not sample['metadatas']:
                return {'total_chunks': count, 'unique_papers': 0}
            
            # Analyze categories and papers
            unique_papers = set()
            categories = set()
            years = set()
            
            for metadata in sample['metadatas']:
                unique_papers.add(metadata['arxiv_id'])
                categories.update(metadata['categories'].split(', '))
                try:
                    year = metadata['published'][:4]
                    years.add(year)
                except:
                    pass
            
            return {
                'total_chunks': count,
                'unique_papers_sample': len(unique_papers),
                'categories': sorted(list(categories)),
                'years': sorted(list(years)),
                'avg_chunks_per_paper': count / len(unique_papers) if unique_papers else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection statistics: {e}")
            return {'error': str(e)}

class QueryProcessor:
    """Processes and enhances user queries"""
    
    def __init__(self):
        pass
    
    def preprocess_query(self, query: str) -> str:
        """
        Clean and enhance user query
        
        Args:
            query: Raw user query
            
        Returns:
            Processed query
        """
        # Basic cleaning
        query = query.strip()
        
        # Expand common abbreviations in academic context
        abbreviations = {
            'ML': 'machine learning',
            'AI': 'artificial intelligence',
            'NLP': 'natural language processing',
            'CV': 'computer vision',
            'RL': 'reinforcement learning',
            'DL': 'deep learning',
            'LLM': 'large language model',
            'CNN': 'convolutional neural network',
            'RNN': 'recurrent neural network',
            'GAN': 'generative adversarial network'
        }
        
        for abbr, full_form in abbreviations.items():
            query = query.replace(f' {abbr} ', f' {full_form} ')
            query = query.replace(f' {abbr.lower()} ', f' {full_form} ')
        
        return query
    
    def extract_search_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract potential filters from query
        
        Args:
            query: User query
            
        Returns:
            Dictionary with extracted filters
        """
        filters = {}
        
        # Year extraction
        import re
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query)
        if years:
            filters['year'] = years[-1]  # Use most recent year mentioned
        
        # Category hints
        category_mapping = {
            'computer vision': ['cs.CV'],
            'natural language': ['cs.CL'],
            'machine learning': ['cs.LG'],
            'artificial intelligence': ['cs.AI'],
            'robotics': ['cs.RO'],
            'neural networks': ['cs.NE'],
        }
        
        query_lower = query.lower()
        suggested_categories = []
        for keyword, cats in category_mapping.items():
            if keyword in query_lower:
                suggested_categories.extend(cats)
        
        if suggested_categories:
            filters['categories'] = list(set(suggested_categories))
        
        return filters

# Convenience functions for common searches
def search_papers(query: str, k: int = 5) -> List[Dict]:
    """Quick search function"""
    retrieval = RetrievalSystem()
    processor = QueryProcessor()
    
    # Process query
    processed_query = processor.preprocess_query(query)
    filters = processor.extract_search_filters(processed_query)
    
    # Search with filters
    results = retrieval.semantic_search(
        processed_query,
        k=k,
        categories=filters.get('categories')
    )
    
    return results

def get_paper_summary(arxiv_id: str) -> Dict:
    """Get comprehensive summary of a paper"""
    retrieval = RetrievalSystem()
    return retrieval.get_paper_context(arxiv_id)

if __name__ == "__main__":
    # Test retrieval system
    retrieval = RetrievalSystem()
    
    # Show collection stats
    stats = retrieval.get_collection_statistics()
    print(f"Collection statistics: {stats}")
    
    # Test search
    if stats.get('total_chunks', 0) > 0:
        results = search_papers("transformer attention mechanism", k=3)
        
        print(f"\nSearch results for 'transformer attention mechanism':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata']['title']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Text preview: {result['text'][:200]}...")
    else:
        print("No papers in collection. Run ingestion first.")