import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, date

try:
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.config import Property, DataType, Configure
    WEAVIATE_AVAILABLE = True
except ImportError as e:
    print(f"Weaviate import error: {e}")
    WEAVIATE_AVAILABLE = False

class WeaviateStore:
    def __init__(self):
        if not WEAVIATE_AVAILABLE:
            print("Weaviate not available - using mock mode")
            self.client = None
            return
            
        try:
            # Load API key and URL from environment variables
            self.api_key = os.environ.get("WEAVIATE_API_KEY")
            self.url = os.environ.get("WEAVIATE_URL")
            
            if not self.api_key or not self.url:
                print("Weaviate credentials not found - using mock mode")
                self.client = None
                return

            # Initialize v4 client (Cloud connection)
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key),
            )
        except Exception as e:
            print(f"Error initializing Weaviate: {e}")
            self.client = None

    def generate_class_name(self, class_name: str = None) -> str:
        """Generate a unique collection name if not provided"""
        return f"Collection_{uuid.uuid4().int}" if class_name is None else f"Collection_{class_name}"

    def create_class_obj(self, class_name: str) -> None:
        """Create a collection (schema) in Weaviate v4"""
        if not self.client:
            print(f"✅ Mock: Created collection: {class_name}")
            return
            
        try:
            self.client.collections.create(
                name=class_name,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="tag", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # disable auto-vectorization
            )
            print(f"✅ Created collection: {class_name}")
        except Exception as e:
            print(f"❌ Error creating collection: {e}")

    def create_chronology_class(self, class_name: str = "ChronologyEvents") -> None:
        """Create a collection specifically for chronology events"""
        if not self.client:
            print(f"✅ Mock: Created chronology collection: {class_name}")
            return
            
        try:
            self.client.collections.create(
                name=class_name,
                properties=[
                    Property(name="event_description", data_type=DataType.TEXT),
                    Property(name="event_type", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),
                    Property(name="normalized_date", data_type=DataType.TEXT),
                    Property(name="confidence_score", data_type=DataType.NUMBER),
                    Property(name="source_text", data_type=DataType.TEXT),
                    Property(name="document_section", data_type=DataType.TEXT),
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="temporal_expressions", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # disable auto-vectorization
            )
            print(f"✅ Created chronology collection: {class_name}")
        except Exception as e:
            print(f"❌ Error creating chronology collection: {e}")

    def add_documents(self, class_name: str, tagged_documents: list) -> None:
        """Insert documents into a given collection"""
        if not self.client:
            print(f"✅ Mock: Added {len(tagged_documents)} documents to {class_name}")
            return
            
        try:
            collection = self.client.collections.get(class_name)
            objects = []
            for d in tagged_documents:
                for doc in d["documents"]:
                    objects.append({
                        "content": doc.page_content,
                        "tag": d["heading"]
                    })
            collection.data.insert_many(objects)
            print(f"✅ Added documents to {class_name}")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")

    def bm25_search_weaviate(self, query: str, class_name: str, limit: int = 4) -> list:
        """Search documents using BM25"""
        if not self.client:
            print(f"✅ Mock: BM25 search for '{query}' in {class_name}")
            return [f"Mock result {i+1} for query: {query}" for i in range(min(limit, 3))]
            
        try:
            collection = self.client.collections.get(class_name)
            result = collection.query.bm25(query=query, limit=limit)
            return [o.properties["content"] for o in result.objects]
        except Exception as e:
            print(f"❌ BM25 search error: {e}")
            return []

    def vector_search_weaviate(self, query_vector: list, class_name: str, limit: int = 4) -> list:
        """Search documents using vector embeddings"""
        if not self.client:
            print(f"✅ Mock: Vector search in {class_name}")
            return [f"Mock vector result {i+1}" for i in range(min(limit, 3))]
            
        try:
            collection = self.client.collections.get(class_name)
            result = collection.query.near_vector(query_vector=query_vector, limit=limit)
            return [o.properties["content"] for o in result.objects]
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []

    def hybrid_search_weaviate(self, query: str, class_name: str, limit: int = 4, alpha: float = 0.5) -> list:
        """Hybrid search = BM25 + Vector"""
        if not self.client:
            print(f"✅ Mock: Hybrid search for '{query}' in {class_name}")
            return [f"Mock hybrid result {i+1} for query: {query}" for i in range(min(limit, 3))]
            
        try:
            collection = self.client.collections.get(class_name)
            result = collection.query.hybrid(query=query, alpha=alpha, limit=limit)
            return [o.properties["content"] for o in result.objects]
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            return []

    def delete_class(self, class_name: str) -> None:
        """Delete a collection and its documents"""
        try:
            self.client.collections.delete(class_name)
            print(f"🗑️ Deleted collection: {class_name}")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")

    # ===== CHRONOLOGY-SPECIFIC METHODS =====

    def add_chronology_events(self, class_name: str, events: List[Dict[str, Any]], document_id: str) -> None:
        """Insert chronology events into a given collection"""
        if not self.client:
            print(f"✅ Mock: Added {len(events)} chronology events to {class_name}")
            return
            
        try:
            collection = self.client.collections.get(class_name)
            objects = []
            
            for event in events:
                # Convert event to Weaviate object format
                obj = {
                    "event_description": event.get("description", ""),
                    "event_type": event.get("event_type", "other"),
                    "timestamp": event.get("timestamp"),
                    "normalized_date": event.get("normalized_date", ""),
                    "confidence_score": event.get("confidence_score", 0.0),
                    "source_text": event.get("source_text", ""),
                    "document_section": event.get("document_section", ""),
                    "document_id": document_id,
                    "temporal_expressions": str(event.get("temporal_expressions", [])),
                    "metadata": str(event.get("metadata", {}))
                }
                objects.append(obj)
            
            collection.data.insert_many(objects)
            print(f"✅ Added {len(objects)} chronology events to {class_name}")
        except Exception as e:
            print(f"❌ Error adding chronology events: {e}")

    def search_chronology_events(self, query: str, class_name: str, 
                                search_type: str = "bm25", limit: int = 10) -> List[Dict[str, Any]]:
        """Search chronology events using different search methods"""
        if not self.client:
            print(f"✅ Mock: Chronology search for '{query}' in {class_name}")
            return [
                {
                    "id": f"mock_{i}",
                    "description": f"Mock event {i+1} for query: {query}",
                    "event_type": "other",
                    "timestamp": None,
                    "normalized_date": "2024-01-01",
                    "confidence_score": 0.8,
                    "source_text": f"Mock source text {i+1}",
                    "document_section": "mock_section",
                    "document_id": "mock_doc",
                    "metadata": "{}"
                }
                for i in range(min(limit, 3))
            ]
            
        try:
            collection = self.client.collections.get(class_name)
            
            if search_type.lower() == "bm25":
                result = collection.query.bm25(query=query, limit=limit)
            elif search_type.lower() == "hybrid":
                result = collection.query.hybrid(query=query, limit=limit)
            else:
                result = collection.query.bm25(query=query, limit=limit)
            
            # Convert results to dictionary format
            events = []
            for obj in result.objects:
                event = {
                    "id": obj.uuid,
                    "description": obj.properties.get("event_description", ""),
                    "event_type": obj.properties.get("event_type", ""),
                    "timestamp": obj.properties.get("timestamp"),
                    "normalized_date": obj.properties.get("normalized_date", ""),
                    "confidence_score": obj.properties.get("confidence_score", 0.0),
                    "source_text": obj.properties.get("source_text", ""),
                    "document_section": obj.properties.get("document_section", ""),
                    "document_id": obj.properties.get("document_id", ""),
                    "metadata": obj.properties.get("metadata", "")
                }
                events.append(event)
            
            return events
        except Exception as e:
            print(f"❌ Error searching chronology events: {e}")
            return []

    def search_events_by_date_range(self, class_name: str, start_date: date, 
                                   end_date: date, limit: int = 50) -> List[Dict[str, Any]]:
        """Search events within a specific date range"""
        if not self.client:
            print(f"✅ Mock: Date range search from {start_date} to {end_date}")
            return [
                {
                    "id": f"mock_date_{i}",
                    "description": f"Mock event {i+1} in date range",
                    "event_type": "other",
                    "timestamp": None,
                    "normalized_date": start_date.isoformat(),
                    "confidence_score": 0.8,
                    "source_text": f"Mock source text {i+1}",
                    "document_section": "mock_section",
                    "document_id": "mock_doc",
                    "metadata": "{}"
                }
                for i in range(min(limit, 3))
            ]
            
        try:
            collection = self.client.collections.get(class_name)
            
            # Use where filter for date range
            result = collection.query.get(
                where={
                    "path": ["normalized_date"],
                    "operator": "And",
                    "operands": [
                        {
                            "path": ["normalized_date"],
                            "operator": "GreaterThanEqual",
                            "valueString": start_date.isoformat()
                        },
                        {
                            "path": ["normalized_date"],
                            "operator": "LessThanEqual",
                            "valueString": end_date.isoformat()
                        }
                    ]
                },
                limit=limit
            )
            
            # Convert results to dictionary format
            events = []
            for obj in result.objects:
                event = {
                    "id": obj.uuid,
                    "description": obj.properties.get("event_description", ""),
                    "event_type": obj.properties.get("event_type", ""),
                    "timestamp": obj.properties.get("timestamp"),
                    "normalized_date": obj.properties.get("normalized_date", ""),
                    "confidence_score": obj.properties.get("confidence_score", 0.0),
                    "source_text": obj.properties.get("source_text", ""),
                    "document_section": obj.properties.get("document_section", ""),
                    "document_id": obj.properties.get("document_id", ""),
                    "metadata": obj.properties.get("metadata", "")
                }
                events.append(event)
            
            return events
        except Exception as e:
            print(f"❌ Error searching events by date range: {e}")
            return []

    def search_events_by_type(self, class_name: str, event_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search events by event type"""
        if not self.client:
            print(f"✅ Mock: Search by event type '{event_type}'")
            return [
                {
                    "id": f"mock_type_{i}",
                    "description": f"Mock {event_type} event {i+1}",
                    "event_type": event_type,
                    "timestamp": None,
                    "normalized_date": "2024-01-01",
                    "confidence_score": 0.8,
                    "source_text": f"Mock source text {i+1}",
                    "document_section": "mock_section",
                    "document_id": "mock_doc",
                    "metadata": "{}"
                }
                for i in range(min(limit, 3))
            ]
            
        try:
            collection = self.client.collections.get(class_name)
            
            result = collection.query.get(
                where={
                    "path": ["event_type"],
                    "operator": "Equal",
                    "valueString": event_type
                },
                limit=limit
            )
            
            # Convert results to dictionary format
            events = []
            for obj in result.objects:
                event = {
                    "id": obj.uuid,
                    "description": obj.properties.get("event_description", ""),
                    "event_type": obj.properties.get("event_type", ""),
                    "timestamp": obj.properties.get("timestamp"),
                    "normalized_date": obj.properties.get("normalized_date", ""),
                    "confidence_score": obj.properties.get("confidence_score", 0.0),
                    "source_text": obj.properties.get("source_text", ""),
                    "document_section": obj.properties.get("document_section", ""),
                    "document_id": obj.properties.get("document_id", ""),
                    "metadata": obj.properties.get("metadata", "")
                }
                events.append(event)
            
            return events
        except Exception as e:
            print(f"❌ Error searching events by type: {e}")
            return []

    def get_chronology_timeline(self, class_name: str, document_id: str) -> List[Dict[str, Any]]:
        """Get all events for a specific document in chronological order"""
        if not self.client:
            print(f"✅ Mock: Get timeline for document '{document_id}'")
            return [
                {
                    "id": f"mock_timeline_{i}",
                    "description": f"Mock timeline event {i+1}",
                    "event_type": "other",
                    "timestamp": None,
                    "normalized_date": f"2024-01-{i+1:02d}",
                    "confidence_score": 0.8,
                    "source_text": f"Mock source text {i+1}",
                    "document_section": "mock_section",
                    "document_id": document_id,
                    "metadata": "{}"
                }
                for i in range(3)
            ]
            
        try:
            collection = self.client.collections.get(class_name)
            
            result = collection.query.get(
                where={
                    "path": ["document_id"],
                    "operator": "Equal",
                    "valueString": document_id
                },
                limit=1000  # Large limit to get all events
            )
            
            # Convert results to dictionary format and sort by date
            events = []
            for obj in result.objects:
                event = {
                    "id": obj.uuid,
                    "description": obj.properties.get("event_description", ""),
                    "event_type": obj.properties.get("event_type", ""),
                    "timestamp": obj.properties.get("timestamp"),
                    "normalized_date": obj.properties.get("normalized_date", ""),
                    "confidence_score": obj.properties.get("confidence_score", 0.0),
                    "source_text": obj.properties.get("source_text", ""),
                    "document_section": obj.properties.get("document_section", ""),
                    "document_id": obj.properties.get("document_id", ""),
                    "metadata": obj.properties.get("metadata", "")
                }
                events.append(event)
            
            # Sort by normalized date
            events.sort(key=lambda x: x.get("normalized_date", ""))
            
            return events
        except Exception as e:
            print(f"❌ Error getting chronology timeline: {e}")
            return []
