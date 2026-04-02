import os
from pypdf import PdfReader
from typing import Dict, Optional

class DocumentParser:
    """
    Handles ingestion of PDF or Text files and simple heuristic-based segmentation
    to feed specific sections to our specialized agents.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_text = ""
        self.sections = {
            "results": "",
            "methods": "",
            "literature_review": "",
            "discussion_conclusion": ""
        }

    def extract_text(self) -> str:
        """Extracts raw text from a PDF or TXT file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        if self.file_path.endswith('.pdf'):
            reader = PdfReader(self.file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            self.raw_text = text
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.raw_text = f.read()
                
        return self.raw_text

    def segment_document(self) -> Dict[str, str]:
        """
        A naive rule-based segmentation of the document based on common academic headers.
        In a production scenario, an LLM router could segment this robustly.
        """
        if not self.raw_text:
            self.extract_text()
            
        # Simplistic segmentation looking for keywords and splitting text.
        text_lower = self.raw_text.lower()
        
        # Find indices of common headers
        methods_idx = text_lower.find('method')
        results_idx = text_lower.find('result')
        discussion_idx = text_lower.find('discussion')
        if discussion_idx == -1:
            discussion_idx = text_lower.find('conclusion')
        lit_review_idx = text_lower.find('literature review')
        if lit_review_idx == -1:
            lit_review_idx = text_lower.find('background')
            
        # Segment based on indices, with fallbacks to quarter chunks if keywords miss.
        length = len(self.raw_text)
        
        if lit_review_idx != -1 and methods_idx != -1 and lit_review_idx < methods_idx:
            self.sections["literature_review"] = self.raw_text[lit_review_idx:methods_idx]
        else:
            self.sections["literature_review"] = self.raw_text[:length//4]
            
        if methods_idx != -1 and results_idx != -1 and methods_idx < results_idx:
            self.sections["methods"] = self.raw_text[methods_idx:results_idx]
        else:
            self.sections["methods"] = self.raw_text[length//4:length//2]
            
        if results_idx != -1 and discussion_idx != -1 and results_idx < discussion_idx:
            self.sections["results"] = self.raw_text[results_idx:discussion_idx]
        else:
            self.sections["results"] = self.raw_text[length//2:3*length//4]
            
        if discussion_idx != -1:
            self.sections["discussion_conclusion"] = self.raw_text[discussion_idx:]
        else:
            self.sections["discussion_conclusion"] = self.raw_text[3*length//4:]
            
        return self.sections
