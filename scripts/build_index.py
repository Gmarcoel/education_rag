import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from scripts.demo import RAGPipeline


def main():
    settings = Settings()
    pipeline = RAGPipeline(settings)
    
    print("Building vector index...")
    result = pipeline.build_index()
    
    print("Index building completed!")
    print(f"Results: {result}")


if __name__ == "__main__":
    main()
