from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from config.settings import Settings


@dataclass
class MockJudgeResult:
    score: int
    rationale: str
    criteria: str


class MockJudge(ABC):
    
    @abstractmethod
    def evaluate(self, question: str, answer: str, reference: str, context: str | None = None) -> MockJudgeResult:
        pass


class MockCorrectnessJudge(MockJudge):
    
    def evaluate(self, question: str, answer: str, reference: str, context: str | None = None) -> MockJudgeResult:
        score = random.randint(3, 5)
        rationale = f"The answer provides {'good' if score >= 4 else 'adequate'} factual information about the question topic."
        return MockJudgeResult(score=score, rationale=rationale, criteria="correctness")


class MockRelevanceJudge(MockJudge):
    
    def evaluate(self, question: str, answer: str, reference: str, context: str | None = None) -> MockJudgeResult:
        score = random.randint(3, 5)
        rationale = f"The question is {'highly' if score >= 4 else 'somewhat'} relevant to programming education."
        return MockJudgeResult(score=score, rationale=rationale, criteria="relevance")


class MockGroundednessJudge(MockJudge):
    
    def evaluate(self, question: str, answer: str, reference: str, context: str | None = None) -> MockJudgeResult:
        if not context:
            return MockJudgeResult(score=1, rationale="No context provided", criteria="groundedness")
        
        score = random.randint(4, 5)
        rationale = f"The answer is {'well' if score >= 4 else 'adequately'} supported by the provided context."
        return MockJudgeResult(score=score, rationale=rationale, criteria="groundedness")


class MockJudgeFactory:
    
    @staticmethod
    def create_judge(judge_type: str, settings: Settings) -> MockJudge:
        judges = {
            "correctness": MockCorrectnessJudge,
            "relevance": MockRelevanceJudge,
            "groundedness": MockGroundednessJudge
        }
        
        if judge_type not in judges:
            raise ValueError(f"Unknown judge type: {judge_type}")
        
        return judges[judge_type]()
