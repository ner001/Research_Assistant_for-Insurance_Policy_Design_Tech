from typing import List
import re


def clean_text(text: str) -> str:
    """
    Lightweight text cleaner to reduce token load by 20-30%
    Removes headers, footers, page numbers, excess whitespace, and formatting artifacts
    """
    # Split into lines and basic cleaning
    lines = [line.strip() for line in text.splitlines() if line.strip()]
        
    # Remove common PDF artifacts
    cleaned_lines = []
    for line in lines:
        # Skip lines that are likely page numbers
        if line.isdigit() or re.match(r'^Page\s+\d+', line, re.IGNORECASE):
            continue
            
        # Skip lines that are just page headers/footers (common patterns)
        if re.match(r'^(Page|Chapter|\d+\s*$)', line, re.IGNORECASE):
            continue
                
        # Skip lines with mostly special characters or formatting
        if len(re.sub(r'[^\w\s]', '', line)) < 3:
            continue
                
        # Skip repeated header/footer patterns
        if line.lower() in ['table of contents', 'appendix', 'references']:
            continue
                
        # Clean up excessive whitespace and special characters
        line = re.sub(r'\s+', ' ', line)  # Multiple spaces to single space
        line = re.sub(r'[^\w\s.,;:!?()-]', '', line)  # Remove unusual special chars
            
        # Skip very short lines that are likely artifacts
        if len(line.split()) >= 3:  # Keep lines with at least 3 words
            cleaned_lines.append(line)
        
    # Join lines and additional cleaning
    cleaned_text = " ".join(cleaned_lines)
        
    # Remove table artifacts and repeated patterns
    cleaned_text = _remove_table_artifacts(cleaned_text)
    cleaned_text = _remove_repeated_patterns(cleaned_text)
        
    # Final cleanup
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
    return cleaned_text
    
def _remove_table_artifacts(text: str) -> str:
    """Remove common table formatting artifacts"""
    # Remove patterns like "| | |" or "--- ---"
    text = re.sub(r'\|[\s\|]*\|', ' ', text)
    text = re.sub(r'[-=]{3,}', ' ', text)
    text = re.sub(r'_{3,}', ' ', text)
    return text

def _remove_repeated_patterns(text: str) -> str:
    """Remove obviously repeated header/footer content"""
    words = text.split()
    if len(words) < 10:
        return text
            
    # Simple approach: if same phrase appears >3 times, it's likely header/footer
    word_counts = {}
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3])
        word_counts[phrase] = word_counts.get(phrase, 0) + 1
    
    # Remove phrases that appear too frequently (likely headers/footers)
    for phrase, count in word_counts.items():
        if count > 3:
            text = text.replace(phrase, " ")
        
    return re.sub(r'\s+', ' ', text).strip()


