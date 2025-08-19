"""
arXiv paper ingestion and text processing
"""
import arxiv
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import requests
from datetime import datetime, timedelta

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivIngestion:
    """Handles fetching and processing arXiv papers"""
    
    def __init__(self):
        self.storage_path = Path(settings.papers_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def search_papers(
        self, 
        query: str, 
        max_results: int = None,
        categories: Optional[List[str]] = None,
        date_from: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Search arXiv papers and return metadata
        
        Args:
            query: Search query string
            max_results: Maximum number of papers to fetch
            categories: List of arXiv categories (e.g., ['cs.AI', 'cs.LG'])
            date_from: Only fetch papers after this date
        """
        max_results = max_results or settings.arxiv_max_results
        
        # Build search query
        search_query = query
        if categories:
            cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({query}) AND ({cat_filter})"
        
        logger.info(f"Searching arXiv: '{search_query}' (max: {max_results})")
        
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in search.results():
            # Date filtering
            if date_from and result.published < date_from:
                continue
                
            paper_data = {
                'arxiv_id': result.entry_id.split('/')[-1],
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'categories': result.categories,
                'published': result.published,
                'updated': result.updated,
                'pdf_url': result.pdf_url,
                'primary_category': result.primary_category
            }
            papers.append(paper_data)
            
            # Rate limiting
            time.sleep(settings.arxiv_delay)
        
        logger.info(f"Found {len(papers)} papers")
        return papers
    
    def download_pdf(self, paper_data: Dict) -> Optional[str]:
        """
        Download PDF for a paper
        
        Args:
            paper_data: Paper metadata dict
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        arxiv_id = paper_data['arxiv_id']
        pdf_path = self.storage_path / f"{arxiv_id}.pdf"
        
        # Skip if already downloaded
        if pdf_path.exists():
            logger.info(f"PDF already exists: {arxiv_id}")
            return str(pdf_path)
        
        try:
            logger.info(f"Downloading PDF: {arxiv_id}")
            response = requests.get(paper_data['pdf_url'], timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Failed to download {arxiv_id}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract clean text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Basic cleaning
                text = self._clean_text(text)
                if text.strip():
                    text_blocks.append(text)
            
            doc.close()
            full_text = "\n\n".join(text_blocks)
            
            # Remove references section (often noisy)
            full_text = self._remove_references_section(full_text)
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def _remove_references_section(self, text: str) -> str:
        """Remove references section which is often noisy for RAG"""
        import re
        
        # Find common reference section patterns
        patterns = [
            r'\n\s*References\s*\n.*$',
            r'\n\s*REFERENCES\s*\n.*$',
            r'\n\s*Bibliography\s*\n.*$',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text
    
    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Process a list of papers: download PDFs and extract text
        
        Args:
            papers: List of paper metadata dicts
            
        Returns:
            List of papers with extracted text
        """
        processed_papers = []
        
        for paper in papers:
            logger.info(f"Processing: {paper['title'][:50]}...")
            
            # Download PDF
            pdf_path = self.download_pdf(paper)
            if not pdf_path:
                continue
            
            # Extract text
            full_text = self.extract_text_from_pdf(pdf_path)
            if not full_text:
                continue
            
            # Add text to paper data
            paper['full_text'] = full_text
            paper['pdf_path'] = pdf_path
            paper['processed_at'] = datetime.now()
            
            processed_papers.append(paper)
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"Successfully processed {len(processed_papers)}/{len(papers)} papers")
        return processed_papers

def fetch_recent_ai_papers(days_back: int = 7) -> List[Dict]:
    """Convenience function to fetch recent AI papers"""
    ingestion = ArxivIngestion()
    
    date_from = datetime.now() - timedelta(days=days_back)
    
    papers = ingestion.search_papers(
        query="artificial intelligence OR machine learning OR deep learning",
        max_results=20,
        categories=['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL'],
        date_from=date_from
    )
    
    return ingestion.process_papers(papers)

if __name__ == "__main__":
    # Test the ingestion pipeline
    papers = fetch_recent_ai_papers(days_back=3)
    print(f"Fetched and processed {len(papers)} papers")
    
    for paper in papers[:3]:
        print(f"\nTitle: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"Text length: {len(paper.get('full_text', ''))} characters")