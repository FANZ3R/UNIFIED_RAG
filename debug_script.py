"""
Fixed Debug script to test components individually
Updated to match your actual .env file variable names
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError as e:
        print(f"❌ Streamlit missing: {e}")
    
    try:
        from openai import OpenAI
        print("✅ OpenAI available")
    except ImportError as e:
        print(f"❌ OpenAI missing: {e}")
    
    try:
        from neo4j import GraphDatabase
        print("✅ Neo4j available")
    except ImportError as e:
        print(f"❌ Neo4j missing: {e}")
    
    try:
        from qdrant_client import QdrantClient
        print("✅ Qdrant client available")
    except ImportError as e:
        print(f"❌ Qdrant client missing: {e}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers available")
    except ImportError as e:
        print(f"❌ SentenceTransformers missing: {e}")

def test_environment():
    """Test environment variables - UPDATED TO MATCH YOUR .env FILE"""
    print("\n🔍 Testing environment variables...")
    
    # Updated to match your actual .env file
    required_vars = [
        'QDRANT_URL',                    # Not VECTOR_QDRANT_URL
        'DEFAULT_COLLECTION_NAME',       # Not VECTOR_COLLECTION_NAME  
        'LOCAL_EMBEDDING_MODEL',         # Not VECTOR_EMBEDDING_MODEL
        'KG_NEO4J_URI',
        'KG_NEO4J_USERNAME',
        'KG_NEO4J_PASSWORD',
        'OPENROUTER_API_KEY'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var or 'KEY' in var:
                print(f"✅ {var}: {'*' * 10}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")

def test_vector_connection():
    """Test Qdrant connection - UPDATED VARIABLE NAMES"""
    print("\n🔍 Testing Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        
        url = os.getenv('QDRANT_URL', 'http://localhost:6333')  # Updated variable name
        client = QdrantClient(url=url)
        
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print(f"✅ Connected to Qdrant at {url}")
        print(f"   Available collections: {collection_names}")
        
        target_collection = os.getenv('DEFAULT_COLLECTION_NAME', 'test_business_data')  # Updated variable name
        if target_collection in collection_names:
            print(f"✅ Target collection '{target_collection}' found")
            return True
        else:
            print(f"❌ Target collection '{target_collection}' not found")
            return False
            
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def test_kg_connection():
    """Test Neo4j connection"""
    print("\n🔍 Testing Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('KG_NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('KG_NEO4J_USERNAME', 'neo4j')
        password = os.getenv('KG_NEO4J_PASSWORD')
        
        if not password:
            print("❌ KG_NEO4J_PASSWORD not set")
            return False
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) as count LIMIT 1")
            count = result.single()["count"]
        
        driver.close()
        
        print(f"✅ Connected to Neo4j at {uri}")
        print(f"   Found {count:,} entities")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_llm_setup():
    """Test LLM setup"""
    print("\n🔍 Testing LLM setup...")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        return False
    
    if not api_key.startswith('sk-or-v1-'):
        print("❌ OPENROUTER_API_KEY doesn't look correct (should start with 'sk-or-v1-')")
        return False
    
    print("✅ OpenRouter API key is set and looks correct")
    
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        print("✅ OpenAI client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ OpenAI client initialization failed: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers specifically"""
    print("\n🔍 Testing SentenceTransformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        print(f"Loading model: {model_name}")
        
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded successfully")
        print(f"   Embedding dimension: {embedding_dim}")
        
        # Test a simple embedding
        test_text = "This is a test."
        embedding = model.encode(test_text)
        
        print(f"✅ Test embedding created: {len(embedding)} dimensions")
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformers test failed: {e}")
        return False

def main():
    print("🚀 FIXED UNIFIED CHATBOT DEBUGGING")
    print("=" * 50)
    
    test_imports()
    test_environment()
    
    vector_ok = test_vector_connection()
    kg_ok = test_kg_connection()
    llm_ok = test_llm_setup()
    st_ok = test_sentence_transformers()
    
    print("\n📊 SUMMARY")
    print("=" * 20)
    print(f"Vector System: {'✅' if vector_ok else '❌'}")
    print(f"Knowledge Graph: {'✅' if kg_ok else '❌'}")
    print(f"LLM Setup: {'✅' if llm_ok else '❌'}")
    print(f"SentenceTransformers: {'✅' if st_ok else '❌'}")
    
    if vector_ok and kg_ok and llm_ok and st_ok:
        print("\n🎉 All systems ready! You can run:")
        print("streamlit run unified_chatbot.py")
    else:
        print("\n🔧 Fix the issues above before running the unified chatbot")
        
        if not vector_ok:
            print("\nVector system fixes:")
            print("- Make sure Qdrant is running: docker-compose up -d")
            print("- Check DEFAULT_COLLECTION_NAME matches your actual collection")
            
        if not kg_ok:
            print("\nKnowledge graph fixes:")
            print("- Make sure Neo4j is running")
            print("- Check KG_NEO4J_PASSWORD is correct")
            print("- Verify your data was imported successfully")
            
        if not st_ok:
            print("\nSentenceTransformers fixes:")
            print("- Install: pip install sentence-transformers")
            print("- Install: pip install qdrant-client")

if __name__ == "__main__":
    main()