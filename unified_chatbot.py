"""
Simple Unified Chatbot
Bridges your existing Vector + Knowledge Graph systems without any modifications
"""

import streamlit as st
import sys
from typing import List, Dict, Any, Optional
import logging
import time
import re
from openai import OpenAI

# Direct imports for existing systems
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configuration - Update these with your actual system details
CONFIG = {
    # Your existing vector system
    "vector": {
        "qdrant_url": "http://localhost:6333",
        "collection_name": "business_dataset",  # Update with your actual collection name
        "embedding_model": "all-MiniLM-L6-v2"
    },
    
    # Your existing knowledge graph system
    "kg": {
        "neo4j_uri": "bolt://localhost:7687",
        "username": "neo4j", 
        "password": "your_secure_password"  # Update with your actual password
    },
    
    # LLM configuration
    "llm": {
        "api_key": "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",
        "model": "meta-llama/llama-3-70b-instruct"
    }
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearcher:
    """Searches your existing Qdrant vector system"""
    
    def __init__(self):
        self.qdrant_client = QdrantClient(url=CONFIG["vector"]["qdrant_url"])
        self.embedding_model = SentenceTransformer(CONFIG["vector"]["embedding_model"])
        self.collection_name = CONFIG["vector"]["collection_name"]
        
        # Verify connection
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name not in collection_names:
                st.error(f"Vector collection '{self.collection_name}' not found!")
                st.error(f"Available collections: {collection_names}")
                st.info("Update the collection_name in CONFIG at the top of the script")
                self.connected = False
            else:
                self.connected = True
                logger.info(f"Connected to vector collection: {self.collection_name}")
        except Exception as e:
            st.error(f"Cannot connect to vector system: {e}")
            st.info("Make sure your Qdrant is running on http://localhost:6333")
            self.connected = False
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search vector database"""
        if not self.connected:
            return []
            
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.3,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'content': result.payload.get('text', ''),
                    'header': result.payload.get('header', 'No header'),
                    'field_name': result.payload.get('field_name', 'unknown'),
                    'score': float(result.score),
                    'source_type': 'vector_search',
                    'record_id': result.payload.get('record_id', -1)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

class KnowledgeGraphSearcher:
    """Searches your existing Neo4j knowledge graph using your chatbot.py queries"""
    
    def __init__(self):
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
            logger.info(f"Connected to knowledge graph: {count} entities")
            
        except Exception as e:
            st.error(f"Cannot connect to knowledge graph: {e}")
            st.info("Make sure your Neo4j is running and check the password in CONFIG")
            self.connected = False
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge graph using your existing chatbot.py query strategies"""
        if not self.connected:
            return []
        
        # Your existing query strategies from chatbot.py
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
            },
            {
                "name": "High Confidence Only",
                "cypher": """
                MATCH (source:Entity)-[r]->(target:Entity)
                WHERE source.text IS NOT NULL 
                  AND target.text IS NOT NULL
                  AND toString(source.text) <> 'NaN'
                  AND toString(target.text) <> 'NaN'
                  AND r.confidence > 0.7
                  AND (toLower(toString(source.text)) CONTAINS toLower($q) 
                       OR toLower(toString(target.text)) CONTAINS toLower($q))
                RETURN source.text AS source_entity, 
                       type(r) AS relationship, 
                       target.text AS target_entity,
                       r.confidence AS confidence
                ORDER BY r.confidence DESC
                LIMIT $limit
                """
            }
        ]
        
        all_results = []
        
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
        
        # Remove duplicates and sort by confidence
        unique_results = []
        seen = set()
        
        for result in sorted(all_results, key=lambda x: x.get('confidence', 0), reverse=True):
            key = result['relationship_text']
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results[:limit]

def classify_query(query: str) -> str:
    """Simple query classification"""
    query_lower = query.lower()
    
    # Keywords that suggest different search strategies
    semantic_keywords = ['similar', 'like', 'related to', 'about', 'find content']
    factual_keywords = ['what is', 'who is', 'define', 'explain', 'how does']
    relational_keywords = ['connected', 'relationship', 'link', 'associated with']
    
    if any(keyword in query_lower for keyword in relational_keywords):
        return 'relational'  # Favor knowledge graph
    elif any(keyword in query_lower for keyword in factual_keywords):
        return 'factual'     # Favor knowledge graph
    elif any(keyword in query_lower for keyword in semantic_keywords):
        return 'semantic'    # Favor vector search
    else:
        return 'hybrid'      # Use both equally

def generate_unified_response(query: str, vector_results: List[Dict], kg_results: List[Dict]) -> str:
    """Generate response using both result types"""
    
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
    
    if not context_parts:
        context_text = "No relevant information found in either system."
    else:
        context_text = "\n".join(context_parts)
    
    # Generate response
    prompt = f"""You are an AI assistant with access to both content similarity search and knowledge graph data about business processes, suppliers, and risk management.

User Question: {query}

Available Information:
{context_text}

Instructions:
1. If content similarity results are available, use them for understanding concepts and finding related information
2. If knowledge graph relationships are available, use them for factual connections and specific relationships
3. Combine insights from both sources when possible
4. Clearly distinguish between content-based matches ("similar content shows...") and factual relationships ("the knowledge graph indicates...")
5. If both sources provide information, synthesize them into a comprehensive answer
6. If no relevant information is found, suggest alternative ways to phrase the question

Provide a clear, informative response:"""
    
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
        return f"I found information from both systems but encountered an error generating the response: {e}"

# Streamlit UI
st.set_page_config(
    page_title="Unified RAG Chatbot", 
    page_icon="🔗",
    layout="wide"
)

st.title("🔗 Unified Vector + Knowledge Graph Chatbot")
st.caption("Queries both your existing vector database and knowledge graph")

# Initialize searchers
@st.cache_resource
def init_searchers():
    return VectorSearcher(), KnowledgeGraphSearcher()

try:
    vector_searcher, kg_searcher = init_searchers()
    
    # Sidebar with system status and controls
    with st.sidebar:
        st.header("System Status")
        
        # Vector system status
        if vector_searcher.connected:
            st.success("✅ Vector System Connected")
            st.text(f"Collection: {CONFIG['vector']['collection_name']}")
        else:
            st.error("❌ Vector System Disconnected")
        
        # KG system status
        if kg_searcher.connected:
            st.success("✅ Knowledge Graph Connected") 
        else:
            st.error("❌ Knowledge Graph Disconnected")
        
        st.divider()
        
        # Search controls
        st.subheader("Search Settings")
        vector_limit = st.slider("Vector Results", 1, 10, 5)
        kg_limit = st.slider("KG Results", 1, 15, 8)
        
        st.subheader("Query Examples")
        st.text("💡 Content similarity:")
        st.text("  'Find similar risk processes'")
        st.text("🔍 Factual:")
        st.text("  'What is supplier compliance?'")
        st.text("🔗 Relationships:")  
        st.text("  'How are vendors connected to audits?'")
        
        # System info
        if st.button("Test Connections"):
            with st.spinner("Testing..."):
                vector_test = vector_searcher.search("test", limit=1)
                kg_test = kg_searcher.search("test", limit=1)
                
                st.write(f"Vector test: {len(vector_test)} results")
                st.write(f"KG test: {len(kg_test)} results")

except Exception as e:
    st.error(f"Failed to initialize systems: {e}")
    st.stop()

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
        # Classify query
        query_type = classify_query(prompt)
        
        with st.spinner(f"Searching both systems (detected: {query_type} query)..."):
            start_time = time.time()
            
            # Search both systems
            vector_results = vector_searcher.search(prompt, limit=vector_limit)
            kg_results = kg_searcher.search(prompt, limit=kg_limit)
            
            search_time = time.time() - start_time
        
        # Show search summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Vector Results", len(vector_results))
        with col2:
            st.metric("KG Results", len(kg_results))
        with col3:
            st.metric("Query Type", query_type)
        with col4:
            st.metric("Search Time", f"{search_time:.2f}s")
        
        # Generate unified response
        with st.spinner("Generating unified response..."):
            response = generate_unified_response(prompt, vector_results, kg_results)
            st.write(response)
        
        # Show detailed results in expandable sections
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
with st.expander("💡 How to use this chatbot"):
    st.markdown("""
    **This chatbot bridges your existing systems:**
    
    1. **Vector Search** finds content similar to your query
    2. **Knowledge Graph** finds factual relationships and connections
    3. **Unified Response** combines insights from both sources
    
    **Query Types:**
    - **Semantic queries**: "Find similar processes" → Emphasizes vector search
    - **Factual queries**: "What is X?" → Emphasizes knowledge graph  
    - **Relationship queries**: "How are X and Y connected?" → Emphasizes knowledge graph
    
    **Troubleshooting:**
    - If no results: Check that both systems are running and accessible
    - If one system fails: The other will still provide results
    - Update CONFIG at the top of this file with your actual system details
    """)

# Footer
st.markdown("---")
st.caption("This chatbot connects to your existing vector database and knowledge graph without modifying them.")