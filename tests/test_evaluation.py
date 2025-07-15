from config.settings import Settings
from src.evaluation.metrics import BLEUMetric, PrecisionAtK, RecallAtK


class TestEvaluation:
    
    def test_bleu_metric(self):
        metric = BLEUMetric()
        
        predicted = "The cat is sitting on the mat"
        reference = "A cat is sitting on a mat"
        
        result = metric.evaluate(predicted, reference)
        
        assert result.metric_name == "BLEU"
        assert 0.0 <= result.score <= 1.0
        assert result.details is not None
    
    def test_precision_at_k(self):
        metric = PrecisionAtK(k=3)
        
        retrieved_docs = [
            {'id': 'doc1', 'content': 'content1'},
            {'id': 'doc2', 'content': 'content2'},
            {'id': 'doc3', 'content': 'content3'}
        ]
        
        relevant_docs = ['doc1', 'doc3']
        
        result = metric.evaluate(retrieved_docs, relevant_docs)
        
        assert result.metric_name == "Precision@3"
        assert 0.0 <= result.score <= 1.0
        assert result.score == 2/3
    
    def test_recall_at_k(self):
        metric = RecallAtK(k=3)
        
        retrieved_docs = [
            {'id': 'doc1', 'content': 'content1'},
            {'id': 'doc2', 'content': 'content2'}
        ]
        
        relevant_docs = ['doc1', 'doc3', 'doc4']
        
        result = metric.evaluate(retrieved_docs, relevant_docs)
        
        assert result.metric_name == "Recall@3"
        assert 0.0 <= result.score <= 1.0
        assert result.score == 1/3
