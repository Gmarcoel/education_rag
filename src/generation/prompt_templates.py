class PromptTemplate:
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


class RAGPromptTemplates:
    
    BASIC_QA = PromptTemplate(
        """You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
    )
    
    INSTRUCTIONAL = PromptTemplate(
        """You are an expert educator. Based on the provided educational materials, answer the student's question clearly and concisely.

Educational Materials:
{context}

Student Question: {question}

Provide a clear explanation that:
1. Directly answers the question
2. Uses examples from the materials when relevant
3. Is appropriate for a student learning this topic

Answer:"""
    )
    
    DETAILED_ANALYSIS = PromptTemplate(
        """You are a knowledgeable assistant. Provide a comprehensive answer based on the context provided.

Context:
{context}

Question: {question}

Please provide a detailed response that:
- Directly addresses the question
- Includes relevant details from the context
- Explains any technical concepts clearly
- Provides examples when helpful

Answer:"""
    )
    
    CONCISE_SUMMARY = PromptTemplate(
        """Based on the provided context, give a brief and direct answer to the question.

Context:
{context}

Question: {question}

Provide a concise answer (1-2 sentences):"""
    )


class PromptBuilder:
    
    def __init__(self):
        self.templates = RAGPromptTemplates()
    
    def build_rag_prompt(self, query: str, documents: list[dict[str, object]], 
                        template_name: str = 'BASIC_QA') -> str:
        context = self._format_context(documents)
        template = getattr(self.templates, template_name)
        return template.format(context=context, question=query)
    
    def _format_context(self, documents: list[dict[str, object]]) -> str:
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            source = doc['metadata'].get('source', 'Unknown')
            
            context_part = f"Document {i} (Source: {source}):\n{content}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def build_custom_prompt(self, template: str, **kwargs) -> str:
        custom_template = PromptTemplate(template)
        return custom_template.format(**kwargs)
