"""
Embedding generation and vector storage using ChromaDB
"""
import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings and vector storage"""
    
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="arxiv_papers",
            metadata={"description": "arXiv paper embeddings"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        logger.info(f"Initialized EmbeddingManager with model: {settings.embedding_model}")
    
    def chunk_paper(self, paper_data: Dict) -> List[Dict]:
        """
        Split paper into chunks for embedding
        
        Args:
            paper_data: Paper metadata with 'full_text' field
            
        Returns:
            List of chunk dictionaries
        """
        full_text = paper_data.get('full_text', '')
        if not full_text:
            logger.warning(f"No full text for paper: {paper_data.get('arxiv_id')}")
            return []
        
        # Combine title and abstract as context
        header_context = f"Title: {paper_data['title']}\n\nAbstract: {paper_data['abstract']}\n\n"
        
        # Split main text into chunks
        text_chunks = self.text_splitter.split_text(full_text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
            
            # Add header context to each chunk for better retrieval
            enhanced_chunk = header_context + chunk_text
            
            chunk_id = f"{paper_data['arxiv_id']}_chunk_{i}"
            
            chunk = {
                'chunk_id': chunk_id,
                'arxiv_id': paper_data['arxiv_id'],
                'chunk_index': i,
                'text': enhanced_chunk,
                'original_text': chunk_text,  # Store original without header
                'title': paper_data['title'],
                'authors': paper_data['authors'],
                'abstract': paper_data['abstract'],
                'categories': paper_data['categories'],
                'published': paper_data['published'].isoformat(),
                'pdf_url': paper_data['pdf_url']
            }
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for paper: {paper_data['arxiv_id']}")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=32
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    def store_paper_chunks(self, paper_data: Dict) -> bool:
        """
        Process paper and store chunks with embeddings
        
        Args:
            paper_data: Paper metadata with full_text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if paper already exists
            arxiv_id = paper_data['arxiv_id']
            existing = self.collection.get(
                where={"arxiv_id": arxiv_id},
                limit=1
            )
            
            if existing['ids']:
                logger.info(f"Paper {arxiv_id} already exists in vector DB")
                return True
            
            # Create chunks
            chunks = self.chunk_paper(paper_data)
            if not chunks:
                logger.warning(f"No chunks created for paper: {arxiv_id}")
                return False
            
            # Generate embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            if not embeddings:
                logger.error(f"No embeddings generated for paper: {arxiv_id}")
                return False
            
            # Prepare data for ChromaDB
            ids = [chunk['chunk_id'] for chunk in chunks]
            documents = [chunk['original_text'] for chunk in chunks]  # Store original text
            metadatas = []
            
            for chunk in chunks:
                metadata = {
                    'arxiv_id': chunk['arxiv_id'],
                    'chunk_index': chunk['chunk_index'],
                    'title': chunk['title'],
                    'authors': ', '.join(chunk['authors'][:3]) if chunk['authors'] else '',
                    'categories': ', '.join(chunk['categories']) if chunk['categories'] else '',
                    'published': chunk['published'],
                    'pdf_url': chunk['pdf_url'],
                    'abstract': chunk['abstract'][:500]  # Truncate for metadata limits
                }
                metadatas.append(metadata)
            
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Stored {len(chunks)} chunks for paper: {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store paper {paper_data.get('arxiv_id')}: {e}")
            return False
    
    def batch_ingest_papers(self, papers: List[Dict]) -> Dict[str, int]:
        """
        Batch process multiple papers
        
        Args:
            papers: List of paper metadata dicts with full_text
            
        Returns:
            Statistics dict
        """
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"Starting batch ingestion of {len(papers)} papers")
        
        for i, paper in enumerate(papers, 1):
            arxiv_id = paper.get('arxiv_id', f'unknown_{i}')
            logger.info(f"Processing paper {i}/{len(papers)}: {arxiv_id}")
            
            if not paper.get('full_text'):
                logger.warning(f"No full text for paper: {arxiv_id}")
                stats['skipped'] += 1
                continue
            
            if self.store_paper_chunks(paper):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Batch ingestion complete: {stats}")
        return stats
    
    def delete_paper(self, arxiv_id: str) -> bool:
        """
        Delete all chunks for a specific paper
        
        Args:
            arxiv_id: arXiv paper ID to delete
            
        Returns:
            True if successful
        """
        try:
            # Get all chunks for this paper
            results = self.collection.get(
                where={"arxiv_id": arxiv_id},
                include=['ids']
            )
            
            if not results['ids']:
                logger.info(f"No chunks found for paper: {arxiv_id}")
                return True
            
            # Delete all chunks
            self.collection.delete(ids=results['ids'])
            
            logger.info(f"Deleted {len(results['ids'])} chunks for paper: {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete paper {arxiv_id}: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all papers)"""
        try:
            self.client.delete_collection("arxiv_papers")
            self.collection = self.client.create_collection(
                name="arxiv_papers",
                metadata={"description": "arXiv paper embeddings"}
            )
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

if __name__ == "__main__":
    # Test embedding system
    em = EmbeddingManager()
    
    # Test with a sample paper
    test_paper = {
        'arxiv_id': 'test_paper',
        'title': 'Test Paper on Transformers',
        'abstract': 'This is a test abstract about transformer models.',
        'authors': ['Test Author'],
        'categories': ['cs.AI'],
        'published': datetime.now(),
        'pdf_url': 'https://example.com/test.pdf',
        'full_text': """
        Introduction
        
        Transformer models have revolutionized natural language processing.
        
        Methodology
        
        We propose a new attention mechanism that improves efficiency.
        
        Results
        
        Our model achieves state-of-the-art performance on several benchmarks.
        
        Conclusion
        
        This work demonstrates the potential of efficient transformer architectures.
        """
    }
    
    # Test chunking and embedding
    success = em.store_paper_chunks(test_paper)
    print(f"Test ingestion successful: {success}")
    
    # Show collection stats
    stats = em.collection.count()
    print(f"Collection now has {stats} chunks")