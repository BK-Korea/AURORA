"""
Test script for query interpretation and correction.

Tests various query variations to ensure LLM-based interpretation works correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.nodes.retriever import RetrieverNode
from src.llm.glm_client import GLMChat
from src.vectorstore.chroma_store import ChromaStore
from src.llm.glm_client import OpenAIEmbeddings
from pydantic import SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

def test_query_correction():
    """Test query correction with various typos and spacing issues."""
    print("=" * 80)
    print("Testing Query Correction (RAG-style)")
    print("=" * 80)
    
    # Initialize LLM
    llm = GLMChat(
        api_key=SecretStr(os.getenv("GLM_API_KEY")),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4.7",
        temperature=0.1,
    )
    
    # Create a dummy vector store for testing
    from unittest.mock import MagicMock
    mock_vector_store = MagicMock()
    
    # Create retriever
    retriever = RetrieverNode(
        vector_store=mock_vector_store,
        llm=llm,
        progress_callback=lambda msg: print(f"  {msg}")
    )
    
    test_cases = [
        "archer의 팔 케 이에서 찾아줘",
        "아처의 팔케이",
        "아 처의 8-k",
        "조비의 십 케 이",
        "joby의 10-q",
    ]
    
    print("\n1. Query Correction Tests:")
    for query in test_cases:
        corrected = retriever.correct_query(query)
        print(f"  원본: {query}")
        print(f"  교정: {corrected}")
        print()

def test_unified_interpretation():
    """Test unified query interpretation."""
    print("=" * 80)
    print("Testing Unified Query Interpretation")
    print("=" * 80)
    
    # Initialize LLM
    llm = GLMChat(
        api_key=SecretStr(os.getenv("GLM_API_KEY")),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4.7",
        temperature=0.1,
    )
    
    # Create a dummy vector store for testing
    from unittest.mock import MagicMock
    mock_vector_store = MagicMock()
    
    # Create retriever
    retriever = RetrieverNode(
        vector_store=mock_vector_store,
        llm=llm,
        progress_callback=lambda msg: print(f"  {msg}")
    )
    
    test_cases = [
        "archer의 팔 케 이에서 찾아줘",
        "아처의 팔케이",
        "아 처의 8-k",
        "조비의 십 케 이",
        "joby의 10-q",
        "ACHR의 8-K",
    ]
    
    print("\n2. Unified Interpretation Tests:")
    for query in test_cases:
        print(f"\n질문: {query}")
        result = retriever.interpret_query_unified(query)
        print(f"  회사/티커: {result.get('company_name_or_ticker')}")
        print(f"  Form Type: {result.get('form_type')}")
        print(f"  날짜: {result.get('date_filter')}")
        print(f"  해석 노트: {result.get('interpretation_notes')}")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AURORA Query Interpretation Test")
    print("=" * 80 + "\n")
    
    try:
        test_query_correction()
        print("\n")
        test_unified_interpretation()
        print("\n" + "=" * 80)
        print("Tests completed!")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
