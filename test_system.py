"""
HealthBot system test - verify all components work.
Run: python test_system.py
"""
import sys

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    try:
        from app.nlp.language_detection import detect_language
        from app.nlp.tokenization import tokenize
        from app.nlp.embeddings import get_embedding_generator
        from app.nlp.pipeline import get_nlp_pipeline
        from app.ml.intent_classifier import IntentClassifier
        from app.ml.symptom_extractor import extract_symptoms
        from app.ml.disease_classifier import DiseaseClassifier
        from app.ml.pipeline import get_ml_pipeline
        from app.dialog.manager import DialogueManager
        from app.knowledge_base.graph import get_disease_info
        from app.llm.ollama_client import call_ollama
        from app.config import DISCLAIMER
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_nlp():
    """Test NLP pipeline."""
    print("\nTesting NLP pipeline...")
    try:
        from app.nlp.pipeline import get_nlp_pipeline
        nlp = get_nlp_pipeline()
        
        # English
        result = nlp.process("I have fever and cough")
        assert result["lang"] == "en"
        assert len(result["tokens"]) > 0
        assert result["embedding"].shape[0] > 0
        print(f"[OK] English: detected '{result['lang']}', tokens: {result['tokens'][:3]}")
        
        # Hindi (skip Unicode print)
        result = nlp.process("mujhe bukhar hai")
        print(f"[OK] Hindi (transliterated): detected '{result['lang']}'")
        
        return True
    except Exception as e:
        print(f"✗ NLP test failed: {e}")
        return False


def test_ml():
    """Test ML pipeline."""
    print("\nTesting ML pipeline...")
    try:
        from app.nlp.pipeline import get_nlp_pipeline
        from app.ml.pipeline import get_ml_pipeline
        
        nlp = get_nlp_pipeline()
        ml = get_ml_pipeline()
        
        text = "I have fever, headache and body pain"
        nlp_out = nlp.process(text)
        ml_out = ml.run(text, nlp_out["tokens"], nlp_out["embedding"])
        
        print(f"✓ Intent: {ml_out['intent']} (confidence: {ml_out['intent_confidence']:.2f})")
        print(f"✓ Symptoms: {ml_out['symptoms'][:3]}")
        print(f"✓ Top diseases: {[d[0] for d in ml_out['top3_diseases']]}")
        print(f"✓ Emergency: {ml_out['is_emergency']}")
        
        return True
    except Exception as e:
        print(f"✗ ML test failed: {e}")
        return False


def test_dialogue():
    """Test dialogue manager."""
    print("\nTesting dialogue manager...")
    try:
        from app.nlp.pipeline import get_nlp_pipeline
        from app.ml.pipeline import get_ml_pipeline
        from app.dialog.manager import DialogueManager
        
        nlp = get_nlp_pipeline()
        ml = get_ml_pipeline()
        dialog = DialogueManager()
        
        text = "I have fever and cough"
        nlp_out = nlp.process(text)
        ml_out = ml.run(text, nlp_out["tokens"], nlp_out["embedding"])
        
        action = dialog.next_action(
            "test_session",
            ml_out["intent"],
            ml_out["symptoms"],
            ml_out["top3_diseases"],
            ml_out["is_emergency"],
            lang=nlp_out["lang"],
        )
        
        response = dialog.build_response(
            "test_session",
            action,
            ml_out["intent"],
            ml_out["symptoms"],
            ml_out["top3_diseases"],
            ml_out["is_emergency"],
            lang=nlp_out["lang"],
            use_ollama_phrasing=False,  # Skip Ollama for test
        )
        
        print(f"✓ Action: {action}")
        print(f"✓ Response preview: {response[:100]}...")
        # Check that response is not empty and contains some meaningful content
        assert len(response) > 0, "Response should not be empty"
        assert "healthcare" in response.lower() or "doctor" in response.lower() or "education" in response.lower() or "prevention" in response.lower() or "advice" in response.lower()
        
        return True
    except Exception as e:
        print(f"✗ Dialogue test failed: {e}")
        return False


def test_knowledge_base():
    """Test knowledge base."""
    print("\nTesting knowledge base...")
    try:
        from app.knowledge_base.graph import get_disease_info, get_prevention
        
        info = get_disease_info("common_cold")
        assert info is not None
        print(f"✓ Disease info: {info['name_en']}")
        
        prevention = get_prevention("flu")
        assert len(prevention) > 0
        print(f"✓ Prevention tips: {prevention[:2]}")
        
        return True
    except Exception as e:
        print(f"✗ Knowledge base test failed: {e}")
        return False


def test_ollama_connection():
    """Test Ollama connection (optional)."""
    print("\nTesting Ollama connection (optional)...")
    try:
        from app.llm.ollama_client import call_ollama
        response = call_ollama("Say 'OK' if you can hear me.", timeout=5)
        if "[Ollama unavailable" in response:
            print("⚠ Ollama not running (responses will use templates only)")
            return True
        else:
            print(f"✓ Ollama connected: {response[:50]}")
            return True
    except Exception as e:
        print(f"⚠ Ollama test skipped: {e}")
        return True


def test_api():
    """Test API can be loaded."""
    print("\nTesting API...")
    try:
        from main import app
        print("✓ FastAPI app loaded")
        return True
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("HealthBot System Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_nlp,
        test_ml,
        test_dialogue,
        test_knowledge_base,
        test_ollama_connection,
        test_api,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    if passed == len(tests):
        print("\n✓ All systems operational!")
        print("\nNext steps:")
        print("  • Run terminal chat: python main.py")
        print("  • Run API server: python main.py --api")
        print("  • Setup WhatsApp/SMS: see DEPLOYMENT.md")
        return 0
    else:
        print("\n⚠ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
