"""
Synthetic dataset generation for RAG evaluation based on HuggingFace cookbook
"""
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from config.settings import Settings
from src.generation.llm import LLMService
from src.storage.database import DatabaseManager, VectorDatabase


@dataclass
class SyntheticQA:
    context: str
    question: str
    answer: str
    source_doc: str
    groundedness_score: int | None = None
    relevance_score: int | None = None
    standalone_score: int | None = None
    groundedness_eval: str | None = None
    relevance_eval: str | None = None
    standalone_eval: str | None = None


class SyntheticDatasetGenerator:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm_service = LLMService(settings)
        
        self.qa_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}
Output:::"""

        self.groundedness_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}
Context: {context}
Answer:::"""

        self.relevance_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to programming education and software development.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer:::"""

        self.standalone_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain technical programming terms and still be a 5: it must simply be clear to a programmer what the question is about.

For instance, "What is the name of the function used in this guide?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer:::"""

    def generate_qa_pair(self, context: str, source_doc: str) -> SyntheticQA | None:
        """Generate a QA pair from a given context"""
        try:
            prompt = self.qa_generation_prompt.format(context=context)
            response = self.llm_service.generate_response(prompt)
            
            # Parse the response
            if "Factoid question:" in response and "Answer:" in response:
                question = response.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
                answer = response.split("Answer: ")[-1].strip()
                
                # Basic validation
                if len(answer) < 300 and len(question) > 10:
                    return SyntheticQA(
                        context=context,
                        question=question,
                        answer=answer,
                        source_doc=source_doc
                    )
        except Exception as e:
            print(f"Error generating QA pair: {e}")
        
        return None

    def critique_question(self, qa: SyntheticQA) -> SyntheticQA:
        """Apply critique agents to evaluate question quality"""
        
        # Groundedness evaluation
        try:
            groundedness_response = self.llm_service.generate_response(
                self.groundedness_prompt.format(context=qa.context, question=qa.question)
            )
            score, evaluation = self._parse_critique_response(groundedness_response)
            qa.groundedness_score = score
            qa.groundedness_eval = evaluation
        except Exception as e:
            print(f"Error in groundedness evaluation: {e}")
        
        # Relevance evaluation
        try:
            relevance_response = self.llm_service.generate_response(
                self.relevance_prompt.format(question=qa.question)
            )
            score, evaluation = self._parse_critique_response(relevance_response)
            qa.relevance_score = score
            qa.relevance_eval = evaluation
        except Exception as e:
            print(f"Error in relevance evaluation: {e}")
        
        # Standalone evaluation
        try:
            standalone_response = self.llm_service.generate_response(
                self.standalone_prompt.format(question=qa.question)
            )
            score, evaluation = self._parse_critique_response(standalone_response)
            qa.standalone_score = score
            qa.standalone_eval = evaluation
        except Exception as e:
            print(f"Error in standalone evaluation: {e}")
        
        return qa

    def _parse_critique_response(self, response: str) -> tuple[int, str]:
        """Parse critique response to extract score and evaluation"""
        try:
            parts = response.split("Total rating: ")
            if len(parts) >= 2:
                score = int(parts[-1].strip().split()[0])
                evaluation = parts[-2].split("Evaluation: ")[-1].strip()
                return score, evaluation
        except:
            pass
        
        return 1, "Failed to parse evaluation"

    def generate_dataset(self, database: VectorDatabase, n_samples: int = 50) -> list[SyntheticQA]:
        """Generate a synthetic evaluation dataset"""
        print(f"Generating {n_samples} synthetic QA pairs...")
        
        # Get documents from database
        documents = database.get_all_documents()
        
        if len(documents) < n_samples:
            print(f"Warning: Only {len(documents)} documents available, generating from all")
            selected_docs = documents
        else:
            selected_docs = random.sample(documents, n_samples)
        
        qa_pairs = []
        
        for doc in selected_docs:
            # Use first 2000 characters as context
            context = doc['content'][:2000]
            source_doc = doc['metadata'].get('filename', 'unknown')
            
            qa = self.generate_qa_pair(context, source_doc)
            if qa:
                qa = self.critique_question(qa)
                qa_pairs.append(qa)
                
                print(f"Generated QA pair {len(qa_pairs)}/{n_samples}")
                if len(qa_pairs) >= n_samples:
                    break
        
        return qa_pairs

    def generate_dataset_with_stats(self, database: VectorDatabase, n_samples: int = 50) -> tuple[list[SyntheticQA], dict]:
        """Generate a synthetic evaluation dataset with detailed statistics"""
        print(f"Generating {n_samples} synthetic QA pairs with statistics...")
        
        stats = {
            "total_documents": 0,
            "by_document": {},
            "avg_scores": {
                "groundedness": 0.0,
                "relevance": 0.0,
                "standalone": 0.0
            },
            "processing_errors": []
        }
        
        # Get documents from database
        documents = database.get_all_documents()
        stats["total_documents"] = len(documents)
        
        if len(documents) < n_samples:
            print(f"Warning: Only {len(documents)} documents available, generating from all")
            selected_docs = documents
        else:
            selected_docs = random.sample(documents, n_samples)
        
        qa_pairs = []
        score_totals = {"groundedness": [], "relevance": [], "standalone": []}
        
        for doc in selected_docs:
            try:
                # Use first 2000 characters as context
                context = doc['content'][:2000]
                source_doc = doc['metadata'].get('filename', 'unknown')
                
                qa = self.generate_qa_pair(context, source_doc)
                if qa:
                    qa = self.critique_question(qa)
                    qa_pairs.append(qa)
                    
                    # Track document statistics
                    stats["by_document"][source_doc] = stats["by_document"].get(source_doc, 0) + 1
                    
                    # Collect scores for averaging
                    if qa.groundedness_score:
                        score_totals["groundedness"].append(qa.groundedness_score)
                    if qa.relevance_score:
                        score_totals["relevance"].append(qa.relevance_score)
                    if qa.standalone_score:
                        score_totals["standalone"].append(qa.standalone_score)
                    
                    print(f"Generated QA pair {len(qa_pairs)}/{n_samples}")
                    if len(qa_pairs) >= n_samples:
                        break
                        
            except Exception as e:
                error_msg = f"Error processing document {doc.get('metadata', {}).get('filename', 'unknown')}: {str(e)}"
                stats["processing_errors"].append(error_msg)
                print(f"Warning: {error_msg}")
        
        # Calculate average scores
        for metric, scores in score_totals.items():
            if scores:
                stats["avg_scores"][metric] = sum(scores) / len(scores)
        
        return qa_pairs, stats

    def filter_dataset(self, qa_pairs: list[SyntheticQA], 
                      min_groundedness: int = 4,
                      min_relevance: int = 4,
                      min_standalone: int = 4) -> list[SyntheticQA]:
        """Filter dataset based on critique scores"""
        filtered = []
        
        for qa in qa_pairs:
            if (qa.groundedness_score and qa.groundedness_score >= min_groundedness and
                qa.relevance_score and qa.relevance_score >= min_relevance and
                qa.standalone_score and qa.standalone_score >= min_standalone):
                filtered.append(qa)
        
        print(f"Filtered {len(filtered)} QA pairs from {len(qa_pairs)} generated")
        return filtered

    def save_dataset(self, qa_pairs: list[SyntheticQA], filepath: str):
        """Save dataset to JSON file"""
        data = []
        for qa in qa_pairs:
            data.append({
                'context': qa.context,
                'question': qa.question,
                'answer': qa.answer,
                'source_doc': qa.source_doc,
                'groundedness_score': qa.groundedness_score,
                'relevance_score': qa.relevance_score,
                'standalone_score': qa.standalone_score,
                'groundedness_eval': qa.groundedness_eval,
                'relevance_eval': qa.relevance_eval,
                'standalone_eval': qa.standalone_eval
            })
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} QA pairs to {filepath}")

    def load_dataset(self, filepath: str) -> list[SyntheticQA]:
        """Load dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = []
        for item in data:
            qa = SyntheticQA(
                context=item['context'],
                question=item['question'],
                answer=item['answer'],
                source_doc=item['source_doc'],
                groundedness_score=item.get('groundedness_score'),
                relevance_score=item.get('relevance_score'),
                standalone_score=item.get('standalone_score'),
                groundedness_eval=item.get('groundedness_eval'),
                relevance_eval=item.get('relevance_eval'),
                standalone_eval=item.get('standalone_eval')
            )
            qa_pairs.append(qa)
        
        print(f"Loaded {len(qa_pairs)} QA pairs from {filepath}")
        return qa_pairs
