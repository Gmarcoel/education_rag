from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time
from pathlib import Path

from config.settings import Settings
from src.generation.llm import LLMService
from src.storage.database import DatabaseManager
from scripts.demo import RAGPipeline


@dataclass
class PerformanceMetrics:
    response_time: float
    input_tokens: int
    output_tokens: int
    api_calls: int
    context_size: int


@dataclass
class AnswerResult:
    answer: str
    context: str
    performance: PerformanceMetrics
    error: Optional[str] = None


class SystemVariant(ABC):
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm_service = LLMService(settings)
    
    @abstractmethod
    def answer_question(self, question: str) -> AnswerResult:
        pass
    
    @abstractmethod
    def get_variant_name(self) -> str:
        pass
    
    def _measure_performance(self, func, input_text: str = "", context: str = "") -> tuple[any, PerformanceMetrics]:
        start_time = time.time()
        result = func()
        end_time = time.time()
        
        input_tokens, output_tokens, total_tokens = self.llm_service.get_last_token_usage()
        
        performance = PerformanceMetrics(
            response_time=end_time - start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_calls=1,
            context_size=len(context)
        )
        
        return result, performance
    
    def _extract_answer_from_result(self, result) -> str:
        if isinstance(result, dict):
            return result.get('answer', '')
        return str(result)


class BaselineLLMVariant(SystemVariant):
    
    def answer_question(self, question: str) -> AnswerResult:
        try:
            def generate_answer():
                prompt = f"""Answer the following question based on your knowledge:

Question: {question}

Provide a clear, concise answer."""
                return self.llm_service.generate_response(prompt)
            
            answer, performance = self._measure_performance(
                generate_answer,
                input_text=question,
                context=""
            )
            
            return AnswerResult(
                answer=answer,
                context="",
                performance=performance
            )
            
        except Exception as e:
            return AnswerResult(
                answer="",
                context="",
                performance=PerformanceMetrics(0, 0, 0, 0, 0),
                error=str(e)
            )
    
    def get_variant_name(self) -> str:
        return "baseline_llm"


class FullContextLLMVariant(SystemVariant):
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._load_all_documents()
    
    def _load_all_documents(self):
        with DatabaseManager(self.settings) as db:
            documents = db.get_all_documents()
            
        self.full_context = "\n\n".join([
            f"Document: {doc['metadata'].get('filename', 'unknown')}\n{doc['content']}"
            for doc in documents
        ])
    
    def answer_question(self, question: str) -> AnswerResult:
        try:
            def generate_answer():
                prompt = f"""Using the information contained in the provided context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, state that clearly.

Context:
{self.full_context}

Question: {question}

Answer:"""
                return self.llm_service.generate_response(prompt)
            
            answer, performance = self._measure_performance(
                generate_answer,
                input_text=question,
                context=self.full_context
            )
            
            return AnswerResult(
                answer=answer,
                context=self.full_context,
                performance=performance
            )
            
        except Exception as e:
            return AnswerResult(
                answer="",
                context=self.full_context,
                performance=PerformanceMetrics(0, 0, 0, 0, len(self.full_context)),
                error=str(e)
            )
    
    def get_variant_name(self) -> str:
        return "full_context_llm"


class RAGSystemVariant(SystemVariant):
    
    def __init__(self, settings: Settings, top_k: int = 5):
        super().__init__(settings)
        self.top_k = top_k
        self.pipeline = RAGPipeline(settings)
        
        # Use the pipeline's LLM service to capture tokens correctly
        self.llm_service = self.pipeline.llm_service
        
        if self.pipeline.get_database_stats()["document_count"] == 0:
            self.pipeline.build_index()
    
    def answer_question(self, question: str) -> AnswerResult:
        try:
            def generate_answer():
                return self.pipeline.query(question, self.top_k)
            
            result, performance = self._measure_performance(
                generate_answer,
                input_text=question,
                context=""
            )
            
            context = self._extract_context_from_result(result)
            
            # Update context size with actual context used
            performance.context_size = len(context)
            
            return AnswerResult(
                answer=result.get('answer', ''),
                context=context,
                performance=performance
            )
            
        except Exception as e:
            return AnswerResult(
                answer="",
                context="",
                performance=PerformanceMetrics(0, 0, 0, 0, 0),
                error=str(e)
            )
    
    def _extract_context_from_result(self, result: dict) -> str:
        if 'context' in result:
            return result['context']
        elif 'documents' in result:
            return "\n".join([doc.get('content', '') for doc in result['documents']])
        return ""
    
    def get_variant_name(self) -> str:
        return f"rag_system_k{self.top_k}"


class SystemVariantFactory:
    
    @staticmethod
    def create_variant(variant_type: str, settings: Settings, **kwargs) -> SystemVariant:
        if variant_type == "baseline":
            return BaselineLLMVariant(settings)
        elif variant_type == "full_context":
            return FullContextLLMVariant(settings)
        elif variant_type == "rag":
            top_k = kwargs.get('top_k', 5)
            return RAGSystemVariant(settings, top_k)
        else:
            raise ValueError(f"Unknown variant type: {variant_type}")
    
    @staticmethod
    def get_available_variants() -> list[str]:
        return ["baseline", "full_context", "rag"]
