import re
from typing import List
from langchain_core.documents import Document


def clean_pdf_text_for_prompt(text: str) -> str:
    """
    Cleans PDF-extracted text for direct LLM ingestion.
    Preserves true paragraph breaks and fixes false ones mid-sentence.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned and formatted text
    """
    # Remove page numbers or running headers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*The World As Will And Idea.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Chapter VIII. On The Theory Of The Ludicrous.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Chapter VIII.20 On The Theory Of The Ludicrous. ', '', text, flags=re.MULTILINE)

    # Remove bracketed page markers like [057]
    text = re.sub(r'\[\d+\]', '', text)

    # Fix hyphenated words that split across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Remove inline citations and references
    text = re.sub(r'\([^)]*[A-Z][a-z]+\.,[^)]*\d+\)', '', text)
    text = re.sub(r'Bk\.\s*[IVXLC]+\.,?\s*ch\.\s*[ivxlc]+\.\s*§\s*\d+', '', text)
    text = re.sub(r'§\s*\d+[A-Za-z. ]*', '', text)
    text = re.sub(r'\([^)]*[\dA-Z][^)]*\)', '', text)
    
    # Collapse multiple spaces left after removal
    text = re.sub(r'\s{2,}', ' ', text)

    # Normalize basic newlines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Fix false double newlines (mid-sentence breaks)
    text = re.sub(r'\n{2,}(?=[a-z,;:])', ' ', text)

    # Collapse multiple paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize spacing
    text = re.sub(r' {2,}', ' ', text)

    # Normalize smart quotes and punctuation
    replacements = {
        '"': '"', '"': '"', ''': "'", ''': "'",
        '—': '-', '–': '-', '…': '...',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Trim paragraphs cleanly
    text = "\n\n".join(p.strip() for p in text.split("\n\n") if p.strip())

    return text.strip()


def join_theory_chunks(docs: List[Document]) -> str:
    """
    Joins cleaned document chunks into a single theory text.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Concatenated theory text
    """
    theory_chunks = [doc.page_content for doc in docs]
    return "\n".join(theory_chunks)
