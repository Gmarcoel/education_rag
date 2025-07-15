from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from config.settings import Settings
from src.generation.llm import LLMService


@dataclass
class JudgeResult:
    score: int
    rationale: str
    criteria: str


class Judge(ABC):
    
    @abstractmethod
    def evaluate(self, question: str, answer: str, reference: str, context: Optional[str] = None) -> JudgeResult:
        pass


class CorrectnessJudge(Judge):
    
    def __init__(self, settings: Settings):
        self.llm_service = LLMService(settings)
        self.prompt_template = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{question}

###Response to evaluate:
{answer}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

    def evaluate(self, question: str, answer: str, reference: str, context: Optional[str] = None) -> JudgeResult:
        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            reference=reference
        )
        
        response = self.llm_service.generate_response(prompt)
        
        try:
            parts = response.split("[RESULT]")
            if len(parts) >= 2:
                feedback = parts[0].replace("Feedback:", "").strip()
                score = int(parts[1].strip().split()[0])
                return JudgeResult(score=score, rationale=feedback, criteria="correctness")
        except Exception:
            pass
        
        return JudgeResult(score=1, rationale="Failed to parse evaluation", criteria="correctness")


class RelevanceJudge(Judge):
    
    def __init__(self, settings: Settings):
        self.llm_service = LLMService(settings)
        self.prompt_template = """###Task Description:
Evaluate how relevant and helpful the response is to the given question in the context of programming education.
1. Write a detailed feedback assessing the relevance of the response to the question.
2. After writing feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The question:
{question}

###Response to evaluate:
{answer}

###Score Rubrics:
[Is the response relevant and helpful for the given question?]
Score 1: The response is completely irrelevant to the question.
Score 2: The response is mostly irrelevant to the question.
Score 3: The response is somewhat relevant to the question.
Score 4: The response is mostly relevant and helpful for the question.
Score 5: The response is completely relevant and very helpful for the question.

###Feedback:"""

    def evaluate(self, question: str, answer: str, reference: str, context: Optional[str] = None) -> JudgeResult:
        prompt = self.prompt_template.format(question=question, answer=answer)
        
        response = self.llm_service.generate_response(prompt)
        
        try:
            parts = response.split("[RESULT]")
            if len(parts) >= 2:
                feedback = parts[0].replace("Feedback:", "").strip()
                score = int(parts[1].strip().split()[0])
                return JudgeResult(score=score, rationale=feedback, criteria="relevance")
        except Exception:
            pass
        
        return JudgeResult(score=1, rationale="Failed to parse evaluation", criteria="relevance")


class GroundednessJudge(Judge):
    
    def __init__(self, settings: Settings):
        self.llm_service = LLMService(settings)
        self.prompt_template = """###Task Description:
Evaluate how well the response is grounded in the provided context. The response should only contain information that can be directly supported by the context.
1. Write a detailed feedback assessing how well the response is supported by the context.
2. After writing feedback, write a score that is an integer between 1 and 5.
3. The output format should look as follows: "Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The question:
{question}

###Context:
{context}

###Response to evaluate:
{answer}

###Score Rubrics:
[Is the response well-grounded in the provided context?]
Score 1: The response contains information not supported by the context or contradicts the context.
Score 2: The response is mostly not grounded in the context.
Score 3: The response is somewhat grounded in the context.
Score 4: The response is mostly grounded in the context.
Score 5: The response is completely grounded in and well-supported by the context.

###Feedback:"""

    def evaluate(self, question: str, answer: str, reference: str, context: Optional[str] = None) -> JudgeResult:
        if not context:
            return JudgeResult(score=1, rationale="No context provided for grounding evaluation", criteria="groundedness")
        
        prompt = self.prompt_template.format(question=question, answer=answer, context=context)
        
        response = self.llm_service.generate_response(prompt)
        
        try:
            parts = response.split("[RESULT]")
            if len(parts) >= 2:
                feedback = parts[0].replace("Feedback:", "").strip()
                score = int(parts[1].strip().split()[0])
                return JudgeResult(score=score, rationale=feedback, criteria="groundedness")
        except Exception:
            pass
        
        return JudgeResult(score=1, rationale="Failed to parse evaluation", criteria="groundedness")


class JudgeFactory:
    
    @staticmethod
    def create_judge(judge_type: str, settings: Settings) -> Judge:
        judges = {
            "correctness": CorrectnessJudge,
            "relevance": RelevanceJudge,
            "groundedness": GroundednessJudge
        }
        
        if judge_type not in judges:
            raise ValueError(f"Unknown judge type: {judge_type}")
        
        return judges[judge_type](settings)
