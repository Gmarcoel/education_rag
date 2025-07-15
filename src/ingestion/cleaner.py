import re


class DocumentCleaner:
    
    def __init__(self):
        self.markdown_patterns = [
            (r'!\[.*?\]\(.*?\)', ''),  # Images
            (r'\[([^\]]+)\]\([^)]+\)', r'\1'),  # Links
            (r'`{3,}[\s\S]*?`{3,}', self._preserve_code_blocks),  # Code blocks
            (r'^\s*#{1,6}\s+', '', re.MULTILINE),  # Headers
            (r'^\s*[-*+]\s+', '', re.MULTILINE),  # Bullet points
            (r'^\s*\d+\.\s+', '', re.MULTILINE),  # Numbered lists
            (r'\*\*(.*?)\*\*', r'\1'),  # Bold
            (r'\*(.*?)\*', r'\1'),  # Italic
            (r'~~(.*?)~~', r'\1'),  # Strikethrough
        ]
    
    def clean(self, document: dict[str, object]) -> dict[str, object]:
        content = document['content']
        
        if document['metadata']['extension'] == '.md':
            content = self._clean_markdown(content)
        
        content = self._clean_whitespace(content)
        content = self._remove_empty_lines(content)
        
        cleaned_document = document.copy()
        cleaned_document['content'] = content
        cleaned_document['metadata']['cleaned'] = True
        
        return cleaned_document
    
    def _clean_markdown(self, content: str) -> str:
        for pattern, replacement, *flags in self.markdown_patterns:
            if callable(replacement):
                content = re.sub(pattern, replacement, content, flags=flags[0] if flags else 0)
            else:
                content = re.sub(pattern, replacement, content, flags=flags[0] if flags else 0)
        return content
    
    def _preserve_code_blocks(self, match) -> str:
        code_block = match.group(0)
        code_content = re.sub(r'`{3,}.*?\n', '', code_block)
        code_content = re.sub(r'`{3,}', '', code_content)
        return f"CODE_BLOCK: {code_content.strip()}"
    
    def _clean_whitespace(self, content: str) -> str:
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def _remove_empty_lines(self, content: str) -> str:
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)
