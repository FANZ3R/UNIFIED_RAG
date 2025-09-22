"""
Fixed Unified Chatbot - Using the exact working configuration from your Qdrant project
"""

import streamlit as st
import os
import sys
from typing import List, Dict, Any, Optional
import logging
import time
import traceback
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables first
load_dotenv()

# Import dependencies with error handling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenAI import failed: {e}")
    OPENAI_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError as e:
    st.error(f"Neo4j import failed: {e}")
    NEO4J_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError as e:
    st.error(f"Qdrant import failed: {e}")
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"SentenceTransformers import failed: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configuration with environment variables - MATCHING YOUR WORKING PROJECT
CONFIG = {
    "vector": {
        "qdrant_url": os.getenv('QDRANT_URL', 'http://localhost:6333'),  # Note: QDRANT_URL not VECTOR_QDRANT_URL
        "collection_name": os.getenv('DEFAULT_COLLECTION_NAME', 'test_business_data'),  # Note: DEFAULT_COLLECTION_NAME
        "embedding_model": os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    },
    "kg": {
        "neo4j_uri": os.getenv('KG_NEO4J_URI', 'bolt://localhost:7687'),
        "username": os.getenv('KG_NEO4J_USERNAME', 'neo4j'),
        "password": os.getenv('KG_NEO4J_PASSWORD', 'your_secure_password')
    },
    "llm": {
        "api_key": os.getenv('OPENROUTER_API_KEY'),
        "model": "meta-llama/llama-3-70b-instruct"
    }
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingModel:
    """Local embedding model using sentence-transformers - EXACT COPY FROM YOUR WORKING PROJECT"""
    
    def __init__(self, model_name: str = None):
        """Initialize local embedding model"""
        if model_name is None:
            model_name = os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def embed_text(self, text):
        """Create embeddings for text"""
        try:
            if isinstance(text, str):
                embedding = self.model.encode(text)
                return embedding.tolist()
            else:
                embeddings = self.model.encode(text)
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dim

class VectorSearcher:
    """Vector searcher using EXACT configuration from your working Qdrant project"""
    
    def __init__(self):
        self.connected = False
        self.error_message = ""
        
        if not QDRANT_AVAILABLE:
            self.error_message = "Qdrant client not available - check installation"
            return
            
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.error_message = "SentenceTransformers not available - check installation"
            return
        
        try:
            # Use EXACT configuration from your working project
            self.qdrant_client = QdrantClient(url=CONFIG["vector"]["qdrant_url"])
            self.collection_name = CONFIG["vector"]["collection_name"]
            
            # Test connection - SAME AS YOUR WORKING PROJECT
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.error_message = f"Collection '{self.collection_name}' not found. Available: {collection_names}"
                return
            
            # Get collection info - SAME AS YOUR WORKING PROJECT
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            self.total_points = collection_info.points_count
            
            # Initialize embedding model - EXACT SAME WAY
            self.embedding_model = LocalEmbeddingModel(CONFIG["vector"]["embedding_model"])
            self.connected = True
            logger.info(f"Vector system connected: {self.collection_name} with {self.total_points:,} points")
            
        except Exception as e:
            self.error_message = f"Vector connection failed: {str(e)}"
            logger.error(f"Vector initialization error: {e}")
            logger.error(traceback.format_exc())
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search vector database - SAME LOGIC AS YOUR WORKING PROJECT"""
        if not self.connected:
            return []
            
        try:
            # Create query embedding - EXACT SAME WAY
            query_embedding = self.embedding_model.embed_text(query)
            
            # Search Qdrant - SAME PARAMETERS
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.3,
                with_payload=True
            )
            
            # Format results - SAME FORMAT AS YOUR WORKING PROJECT
            results = []
            for result in search_results:
                results.append({
                    'content': result.payload.get('text', ''),
                    'header': result.payload.get('header', 'No header'),
                    'field_name': result.payload.get('field_name', 'unknown'),
                    'score': float(result.score),
                    'source_type': 'vector_search',
                    'record_id': result.payload.get('record_id', -1),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'total_chunks': result.payload.get('total_chunks', 1),
                    'source_file': result.payload.get('source_file', ''),
                    'chunk_length': result.payload.get('chunk_length', 0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

class KnowledgeGraphSearcher:
    """Knowledge Graph searcher - UNCHANGED FROM ORIGINAL"""
    
    def __init__(self):
        self.connected = False
        self.error_message = ""
        self.driver = None
        
        if not NEO4J_AVAILABLE:
            self.error_message = "Neo4j driver not available - check installation"
            return
        
        try:
            self.driver = GraphDatabase.driver(
                CONFIG["kg"]["neo4j_uri"], 
                auth=(CONFIG["kg"]["username"], CONFIG["kg"]["password"])
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN count(n) as count LIMIT 1")
                count = result.single()["count"]
            
            self.connected = True
            logger.info(f"KG system connected: {count} entities")
            
        except Exception as e:
            self.error_message = f"Neo4j connection failed: {str(e)}"
            logger.error(f"KG initialization error: {e}")
    
    def __del__(self):
        """Clean up driver connection"""
        if self.driver:
            try:
                self.driver.close()
            except:
                pass
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge graph"""
        if not self.connected:
            return []
        
        queries = [
            {
                "name": "Direct Entity Search",
                "cypher": """
                MATCH (source:Entity)-[r]->(target:Entity)
                WHERE source.text IS NOT NULL 
                  AND target.text IS NOT NULL
                  AND toString(source.text) <> 'NaN'
                  AND toString(target.text) <> 'NaN'
                  AND (toLower(toString(source.text)) CONTAINS toLower($q) 
                       OR toLower(toString(target.text)) CONTAINS toLower($q))
                RETURN source.text AS source_entity, 
                       type(r) AS relationship, 
                       target.text AS target_entity,
                       r.confidence AS confidence,
                       source.label AS source_type,
                       target.label AS target_type
                ORDER BY r.confidence DESC
                LIMIT $limit
                """
            }
        ]
        
        all_results = []
        
        try:
            with self.driver.session() as session:
                for query_info in queries:
                    try:
                        results = session.run(query_info["cypher"], q=query, limit=limit)
                        
                        for record in results:
                            result = {
                                'relationship_text': f"{record['source_entity']} --[{record['relationship']}]--> {record['target_entity']}",
                                'source_entity': record['source_entity'],
                                'target_entity': record['target_entity'],
                                'relationship_type': record['relationship'],
                                'confidence': record.get('confidence', 0.0),
                                'source_type': 'knowledge_graph',
                                'strategy': query_info['name']
                            }
                            all_results.append(result)
                            
                    except Exception as e:
                        logger.warning(f"KG query '{query_info['name']}' failed: {e}")
                        continue
        except Exception as e:
            logger.error(f"KG search session failed: {e}")
            return []
        
        # Remove duplicates and sort by confidence
        unique_results = []
        seen = set()
        
        for result in sorted(all_results, key=lambda x: x.get('confidence', 0), reverse=True):
            key = result['relationship_text']
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results[:limit]

def generate_unified_response(query: str, vector_results: List[Dict], kg_results: List[Dict]) -> str:
    """Generate response using both result types"""
    
    if not OPENAI_AVAILABLE or not CONFIG["llm"]["api_key"]:
        # Fallback response without LLM
        response_parts = [f"Query: {query}\n"]
        
        if vector_results:
            response_parts.append("=== Content Search Results ===")
            for i, result in enumerate(vector_results[:3], 1):
                response_parts.append(f"[{i}] {result['header']} (Score: {result['score']:.3f})")
                response_parts.append(f"    {result['content'][:200]}...\n")
        
        if kg_results:
            response_parts.append("=== Knowledge Graph Results ===")
            for i, result in enumerate(kg_results[:3], 1):
                response_parts.append(f"[{i}] {result['relationship_text']} (Confidence: {result['confidence']:.3f})\n")
        
        if not vector_results and not kg_results:
            response_parts.append("No relevant information found in either system.")
        
        return "\n".join(response_parts)
    
    # Format context for LLM
    context_parts = []
    
    if vector_results:
        context_parts.append("=== CONTENT SIMILARITY RESULTS ===")
        for i, result in enumerate(vector_results[:5], 1):
            context_parts.append(f"[Content {i}] Score: {result['score']:.3f}")
            context_parts.append(f"Field: {result['field_name']}")
            context_parts.append(f"Header: {result['header']}")
            context_parts.append(f"Content: {result['content'][:400]}...")
            context_parts.append("")
    
    if kg_results:
        context_parts.append("=== KNOWLEDGE GRAPH RELATIONSHIPS ===")
        for i, result in enumerate(kg_results[:5], 1):
            context_parts.append(f"[Relationship {i}] Confidence: {result['confidence']:.3f}")
            context_parts.append(f"Connection: {result['relationship_text']}")
            context_parts.append("")
    
    context_text = "\n".join(context_parts) if context_parts else "No relevant information found."
    
    prompt = f"""You are an AI assistant with access to both content similarity search and knowledge graph data.

User Question: {query}

Available Information:
{context_text}

Instructions:
1. Use content similarity results for understanding concepts
2. Use knowledge graph relationships for factual connections
3. Provide a clear, informative response combining both sources when available

Provide a clear response:"""
    
    try:
        client = OpenAI(
            api_key=CONFIG["llm"]["api_key"],
            base_url="https://openrouter.ai/api/v1"
        )
        
        response = client.chat.completions.create(
            model=CONFIG["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"LLM response generation failed: {e}")
        return f"I found information but encountered an error generating the response: {e}"

# Streamlit UI
st.set_page_config(
    page_title="Fixed Unified RAG Chatbot", 
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Fixed Unified Vector + Knowledge Graph Chatbot")
st.caption("Using exact configuration from your working Qdrant project")

# Debug information
with st.expander("🔧 Debug Information - Environment Variables", expanded=False):
    st.write("**Vector System Environment Variables:**")
    st.code(f"QDRANT_URL: {os.getenv('QDRANT_URL', 'Not set')}")
    st.code(f"DEFAULT_COLLECTION_NAME: {os.getenv('DEFAULT_COLLECTION_NAME', 'Not set')}")
    st.code(f"LOCAL_EMBEDDING_MODEL: {os.getenv('LOCAL_EMBEDDING_MODEL', 'Not set')}")
    
    st.write("**Knowledge Graph Environment Variables:**")
    st.code(f"KG_NEO4J_URI: {os.getenv('KG_NEO4J_URI', 'Not set')}")
    st.code(f"KG_NEO4J_USERNAME: {os.getenv('KG_NEO4J_USERNAME', 'Not set')}")
    
    st.write("**Dependencies:**")
    st.write(f"OpenAI: {OPENAI_AVAILABLE}")
    st.write(f"Neo4j: {NEO4J_AVAILABLE}")
    st.write(f"Qdrant: {QDRANT_AVAILABLE}")
    st.write(f"SentenceTransformers: {SENTENCE_TRANSFORMERS_AVAILABLE}")

# Initialize searchers
@st.cache_resource
def init_searchers():
    try:
        vector_searcher = VectorSearcher()
        kg_searcher = KnowledgeGraphSearcher()
        return vector_searcher, kg_searcher
    except Exception as e:
        st.error(f"Failed to initialize searchers: {e}")
        st.error(traceback.format_exc())
        return None, None

vector_searcher, kg_searcher = init_searchers()

if vector_searcher is None or kg_searcher is None:
    st.error("Failed to initialize systems. Check the debug information above.")
    st.stop()

# Sidebar with system status
with st.sidebar:
    st.header("System Status")
    
    # Vector system status
    if vector_searcher.connected:
        st.success("✅ Vector System Connected")
        st.text(f"Collection: {CONFIG['vector']['collection_name']}")
        if hasattr(vector_searcher, 'total_points'):
            st.text(f"Points: {vector_searcher.total_points:,}")
    else:
        st.error("❌ Vector System Disconnected")
        if vector_searcher.error_message:
            st.error(vector_searcher.error_message)
    
    # KG system status
    if kg_searcher.connected:
        st.success("✅ Knowledge Graph Connected") 
    else:
        st.error("❌ Knowledge Graph Disconnected")
        if kg_searcher.error_message:
            st.error(kg_searcher.error_message)
    
    # LLM status
    if CONFIG["llm"]["api_key"]:
        st.success("✅ LLM API Key Set")
    else:
        st.warning("⚠️ LLM API Key Missing (will use fallback)")
    
    st.divider()
    
    # Search controls
    st.subheader("Search Settings")
    vector_limit = st.slider("Vector Results", 1, 10, 5)
    kg_limit = st.slider("KG Results", 1, 15, 8)
    
    # Test connections button
    if st.button("🔄 Test Connections"):
        with st.spinner("Testing connections..."):
            # Test vector
            if vector_searcher.connected:
                try:
                    test_results = vector_searcher.search("test query", limit=1)
                    st.success(f"✅ Vector test: Found {len(test_results)} results")
                except Exception as e:
                    st.error(f"❌ Vector test failed: {e}")
            
            # Test KG
            if kg_searcher.connected:
                try:
                    test_results = kg_searcher.search("test", limit=1)
                    st.success(f"✅ KG test: Found {len(test_results)} results")
                except Exception as e:
                    st.error(f"❌ KG test failed: {e}")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your business data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        # Check if at least one system is connected
        if not vector_searcher.connected and not kg_searcher.connected:
            st.error("Neither system is connected. Please check your configuration.")
            response = "I'm unable to search either system. Please check the system status in the sidebar."
        else:
            with st.spinner("Searching available systems..."):
                start_time = time.time()
                
                # Search both systems (if connected)
                vector_results = []
                kg_results = []
                
                if vector_searcher.connected:
                    try:
                        vector_results = vector_searcher.search(prompt, limit=vector_limit)
                        st.info(f"🔍 Vector search: {len(vector_results)} results")
                    except Exception as e:
                        st.error(f"Vector search failed: {e}")
                
                if kg_searcher.connected:
                    try:
                        kg_results = kg_searcher.search(prompt, limit=kg_limit)
                        st.info(f"📊 KG search: {len(kg_results)} results")
                    except Exception as e:
                        st.error(f"KG search failed: {e}")
                
                search_time = time.time() - start_time
            
            # Show search summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vector Results", len(vector_results))
            with col2:
                st.metric("KG Results", len(kg_results))
            with col3:
                st.metric("Search Time", f"{search_time:.2f}s")
            
            # Generate unified response
            with st.spinner("Generating response..."):
                response = generate_unified_response(prompt, vector_results, kg_results)
                st.write(response)
            
            # Show detailed results in expandable sections
            if vector_results or kg_results:
                with st.expander("🔍 View detailed search results"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Vector Search Results")
                        if vector_results:
                            for i, result in enumerate(vector_results, 1):
                                st.write(f"**Result {i}** (Score: {result['score']:.3f})")
                                st.write(f"*Field:* {result['field_name']}")
                                st.write(f"*Header:* {result['header'][:100]}...")
                                st.write(f"*Content:* {result['content'][:200]}...")
                                if 'record_id' in result:
                                    st.write(f"*Record:* {result['record_id']}")
                                st.divider()
                        else:
                            st.info("No vector results found")
                    
                    with col2:
                        st.subheader("Knowledge Graph Results")
                        if kg_results:
                            for i, result in enumerate(kg_results, 1):
                                st.write(f"**Result {i}** (Confidence: {result['confidence']:.3f})")
                                st.write(f"*Relationship:* {result['relationship_text']}")
                                st.write(f"*Strategy:* {result['strategy']}")
                                st.divider()
                        else:
                            st.info("No knowledge graph results found")
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Instructions
with st.expander("💡 Configuration Guide"):
    st.markdown("""
    **Your .env file should have these exact variables (from your working Qdrant project):**
    
    ```bash
    # Vector System (note the variable names!)
    QDRANT_URL=http://localhost:6333
    DEFAULT_COLLECTION_NAME=test_business_data
    LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
    
    # Knowledge Graph System  
    KG_NEO4J_URI=bolt://localhost:7687
    KG_NEO4J_USERNAME=neo4j
    KG_NEO4J_PASSWORD=your_secure_password
    
    # LLM Configuration
    OPENROUTER_API_KEY=your_api_key
    
    # Other settings from your working project
    CHUNK_SIZE=500
    CHUNK_OVERLAP=50
    BATCH_SIZE=32
    ```
    
    **Key differences fixed:**
    - Using `QDRANT_URL` instead of `VECTOR_QDRANT_URL`
    - Using `DEFAULT_COLLECTION_NAME` instead of `VECTOR_COLLECTION_NAME`
    - Using the exact same embedding model initialization as your working project
    - Using the same Qdrant client configuration
    """)

st.markdown("---")
st.caption("Fixed version using exact configuration from your working Qdrant project")