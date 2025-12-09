from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from config.settings import PDF_PATH, THEORY_START_PAGE, THEORY_END_PAGE
from src.utils.text_processing import clean_pdf_text_for_prompt


class TheoryLoader:
    """Handles loading and preprocessing of the comedy theory document."""
    
    def __init__(self, pdf_path: str = None):
        self.pdf_path = pdf_path or str(PDF_PATH)
        self.docs = None
        self.theory_docs = None
        
    def load_documents(self) -> List[Document]:
        """Load all documents from PDF."""
        loader = PyMuPDFLoader(self.pdf_path)
        self.docs = loader.load()
        
        # Add page metadata
        for i, doc in enumerate(self.docs):
            doc.metadata["page"] = i
            
        return self.docs
    
    def extract_theory_section(self, 
                               start_page: int = THEORY_START_PAGE,
                               end_page: int = THEORY_END_PAGE) -> List[Document]:
        """
        Extract and clean the relevant theory section.
        
        Args:
            start_page: Starting page index
            end_page: Ending page index
            
        Returns:
            List of cleaned theory documents
        """
        if self.docs is None:
            self.load_documents()
            
        self.theory_docs = self.docs[start_page:end_page]
        
        # Clean each document
        for doc in self.theory_docs:
            doc.page_content = clean_pdf_text_for_prompt(doc.page_content)
            
        return self.theory_docs
    
    def get_full_theory_text(self) -> str:
        """Get the complete theory text as a single string."""
        if self.theory_docs is None:
            self.extract_theory_section()
            
        from src.utils.text_processing import join_theory_chunks
        return join_theory_chunks(self.theory_docs)