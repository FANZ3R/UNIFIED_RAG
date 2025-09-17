"""Test connections to your existing systems"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

load_dotenv()

def test_vector_system():
    print("Testing Vector System (Qdrant)...")
    try:
        client = QdrantClient(url=os.getenv('VECTOR_QDRANT_URL'))
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        collection_name = os.getenv('VECTOR_COLLECTION_NAME')
        
        print(f"✅ Connected to Qdrant")
        print(f"   Available: {collection_names}")
        
        if collection_name in collection_names:
            print(f"✅ Found collection: {collection_name}")
            return True
        else:
            print(f"❌ Collection '{collection_name}' not found!")
            return False
    except Exception as e:
        print(f"❌ Vector connection failed: {e}")
        return False

def test_kg_system():
    print("\nTesting Knowledge Graph (Neo4j)...")
    try:
        driver = GraphDatabase.driver(
            os.getenv('KG_NEO4J_URI'), 
            auth=(os.getenv('KG_NEO4J_USERNAME'), os.getenv('KG_NEO4J_PASSWORD'))
        )
        with driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            count = result.single()["count"]
        
        print(f"✅ Connected to Neo4j")
        print(f"   Entities: {count:,}")
        driver.close()
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🔗 TESTING EXISTING SYSTEMS")
    print("=" * 40)
    vector_ok = test_vector_system()
    kg_ok = test_kg_system()
    
    if vector_ok and kg_ok:
        print("\n✅ Ready to run: streamlit run unified_chatbot.py")
    else:
        print("\n❌ Fix connection issues first")