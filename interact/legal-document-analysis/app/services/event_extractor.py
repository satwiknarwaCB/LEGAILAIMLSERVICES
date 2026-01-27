import re
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from app.models.chronology import Event, EventType, TemporalExpression
from app.services.temporal_processor import TemporalProcessor

class EventExtractor:
    """Extracts and classifies events from text using NLP and pattern matching"""
    
    def __init__(self):
        self.temporal_processor = TemporalProcessor()
        
        # Event trigger patterns
        self.event_patterns = {
            EventType.LEGAL_ACTION: [
                r'\b(?:filed|filing|sued|lawsuit|litigation|court|court order|judgment|ruling|appeal|settlement)\b',
                r'\b(?:legal action|legal proceeding|legal matter|dispute|arbitration|mediation)\b',
                r'\b(?:plaintiff|defendant|petitioner|respondent)\b'
            ],
            EventType.CONTRACT_EVENT: [
                r'\b(?:signed|executed|amended|modified|terminated|expired|renewed|extended)\b',
                r'\b(?:contract|agreement|lease|license|permit|warranty|guarantee)\b',
                r'\b(?:effective date|commencement|expiration|renewal)\b'
            ],
            EventType.PAYMENT: [
                r'\b(?:paid|payment|invoice|billing|due date|overdue|refund|reimbursement)\b',
                r'\b(?:\$[\d,]+\.?\d*|USD|dollars?|amount|fee|cost|price|charge)\b',
                r'\b(?:installment|monthly|quarterly|annual|one-time)\b'
            ],
            EventType.DEADLINE: [
                r'\b(?:deadline|due date|expiration|expires|must be completed by|by the end of)\b',
                r'\b(?:no later than|not later than|on or before|prior to)\b',
                r'\b(?:timeline|schedule|milestone|target date)\b'
            ],
            EventType.MEETING: [
                r'\b(?:meeting|conference|call|appointment|session|gathering|assembly)\b',
                r'\b(?:scheduled|planned|arranged|organized|convened)\b',
                r'\b(?:board meeting|committee|review|discussion)\b'
            ],
            EventType.COMMUNICATION: [
                r'\b(?:sent|received|emailed|called|notified|informed|contacted)\b',
                r'\b(?:letter|email|memo|notice|announcement|correspondence)\b',
                r'\b(?:response|reply|acknowledgment|confirmation)\b'
            ],
            EventType.DOCUMENT_CREATION: [
                r'\b(?:created|drafted|prepared|generated|issued|published|released)\b',
                r'\b(?:document|report|statement|certificate|license|permit)\b',
                r'\b(?:version|revision|update|amendment|supplement)\b'
            ],
            EventType.DECISION: [
                r'\b(?:decided|determined|resolved|approved|rejected|denied|granted)\b',
                r'\b(?:decision|resolution|determination|ruling|verdict)\b',
                r'\b(?:board|committee|management|authority|official)\b'
            ],
            EventType.NOTIFICATION: [
                r'\b(?:notified|informed|alerted|warned|advised|announced)\b',
                r'\b(?:notice|alert|warning|advisory|announcement)\b',
                r'\b(?:urgent|immediate|as soon as possible|ASAP)\b'
            ]
        }
        
        # Event context patterns (help identify event boundaries)
        self.context_patterns = [
            r'\b(?:on|at|in|during|after|before|following|preceding)\s+[^.]*',
            r'\b(?:the|a|an)\s+[^.]*',
            r'\b(?:will|shall|must|should|could|would|may|might)\s+[^.]*',
            r'\b(?:has|have|had|was|were|is|are|been)\s+[^.]*'
        ]

    def identify_events(self, text: str, document_section: str = "unknown") -> List[Event]:
        """Identify events in text and extract relevant information"""
        events = []
        
        # Split text into sentences for better event detection
        sentences = self._split_into_sentences(text)
        
        for sentence_idx, sentence in enumerate(sentences):
            # Extract temporal expressions from the sentence
            temporal_expressions = self.temporal_processor.extract_temporal_expressions(sentence)
            
            # Check if sentence contains event indicators
            event_types = self._classify_event_types(sentence)
            
            if event_types or temporal_expressions:
                # Create event
                event = Event(
                    id=str(uuid.uuid4()),
                    description=sentence.strip(),
                    event_type=event_types[0] if event_types else EventType.OTHER,
                    source_text=sentence.strip(),
                    document_section=document_section,
                    temporal_expressions=temporal_expressions,
                    confidence_score=self._calculate_event_confidence(sentence, event_types, temporal_expressions),
                    metadata={
                        'sentence_index': sentence_idx,
                        'detected_types': [t.value for t in event_types],
                        'temporal_count': len(temporal_expressions)
                    }
                )
                events.append(event)
        
        return events

    def classify_event_types(self, events: List[Event]) -> List[Event]:
        """Classify and refine event types using additional context"""
        classified_events = []
        
        for event in events:
            # Re-classify with more context
            refined_types = self._classify_event_types(event.description)
            
            if refined_types:
                event.event_type = refined_types[0]
                event.metadata['refined_types'] = [t.value for t in refined_types]
            
            # Update confidence based on refined classification
            event.confidence_score = self._calculate_event_confidence(
                event.description, refined_types, event.temporal_expressions
            )
            
            classified_events.append(event)
        
        return classified_events

    def extract_event_relationships(self, events: List[Event]) -> List[Dict[str, any]]:
        """Extract relationships between events"""
        relationships = []
        
        # Sort events by position in document
        sorted_events = sorted(events, key=lambda x: x.metadata.get('sentence_index', 0))
        
        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            # Check for causal relationships
            if self._has_causal_relationship(current_event, next_event):
                relationships.append({
                    'type': 'causal',
                    'source': current_event.id,
                    'target': next_event.id,
                    'confidence': 0.7
                })
            
            # Check for temporal relationships
            if self._has_temporal_relationship(current_event, next_event):
                relationships.append({
                    'type': 'temporal',
                    'source': current_event.id,
                    'target': next_event.id,
                    'confidence': 0.8
                })
            
            # Check for same entity relationships
            if self._has_entity_relationship(current_event, next_event):
                relationships.append({
                    'type': 'entity',
                    'source': current_event.id,
                    'target': next_event.id,
                    'confidence': 0.6
                })
        
        return relationships

    def calculate_event_confidence(self, events: List[Event]) -> List[Event]:
        """Calculate and update confidence scores for events"""
        for event in events:
            # Base confidence from pattern matching
            pattern_confidence = self._calculate_pattern_confidence(event.description, event.event_type)
            
            # Temporal confidence
            temporal_confidence = 0.0
            if event.temporal_expressions:
                temporal_confidence = sum(expr.confidence for expr in event.temporal_expressions) / len(event.temporal_expressions)
            
            # Context confidence
            context_confidence = self._calculate_context_confidence(event.description)
            
            # Combined confidence (weighted average)
            event.confidence_score = (
                pattern_confidence * 0.4 +
                temporal_confidence * 0.4 +
                context_confidence * 0.2
            )
        
        return events

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be enhanced with spaCy)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _classify_event_types(self, text: str) -> List[EventType]:
        """Classify event types based on text patterns"""
        text_lower = text.lower()
        detected_types = []
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_types.append(event_type)
                    break  # Only add each type once
        
        return detected_types

    def _calculate_event_confidence(self, text: str, event_types: List[EventType], 
                                  temporal_expressions: List[TemporalExpression]) -> float:
        """Calculate confidence score for an event"""
        confidence = 0.0
        
        # Pattern matching confidence
        if event_types:
            confidence += 0.4
        
        # Temporal expression confidence
        if temporal_expressions:
            temp_confidence = sum(expr.confidence for expr in temporal_expressions) / len(temporal_expressions)
            confidence += temp_confidence * 0.4
        
        # Context confidence
        context_confidence = self._calculate_context_confidence(text)
        confidence += context_confidence * 0.2
        
        return min(1.0, confidence)

    def _calculate_pattern_confidence(self, text: str, event_type: EventType) -> float:
        """Calculate confidence based on pattern matching"""
        if event_type not in self.event_patterns:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        total_patterns = len(self.event_patterns[event_type])
        
        for pattern in self.event_patterns[event_type]:
            if re.search(pattern, text_lower):
                matches += 1
        
        return matches / total_patterns if total_patterns > 0 else 0.0

    def _calculate_context_confidence(self, text: str) -> float:
        """Calculate confidence based on context indicators"""
        confidence = 0.0
        text_lower = text.lower()
        
        # Check for action verbs
        action_verbs = ['filed', 'signed', 'paid', 'sent', 'received', 'created', 'decided', 'notified']
        if any(verb in text_lower for verb in action_verbs):
            confidence += 0.3
        
        # Check for temporal indicators
        temporal_words = ['on', 'at', 'in', 'during', 'after', 'before', 'when', 'while']
        if any(word in text_lower for word in temporal_words):
            confidence += 0.2
        
        # Check for specific entities (names, places, organizations)
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text):  # Proper nouns
            confidence += 0.2
        
        # Check for numbers (dates, amounts, etc.)
        if re.search(r'\b\d+\b', text):
            confidence += 0.1
        
        return min(1.0, confidence)

    def _has_causal_relationship(self, event1: Event, event2: Event) -> bool:
        """Check if two events have a causal relationship"""
        causal_indicators = ['because', 'due to', 'as a result', 'consequently', 'therefore', 'thus', 'hence']
        
        # Check if event2 mentions event1 or vice versa
        text1_lower = event1.description.lower()
        text2_lower = event2.description.lower()
        
        # Simple causal relationship detection
        for indicator in causal_indicators:
            if indicator in text2_lower:
                return True
        
        return False

    def _has_temporal_relationship(self, event1: Event, event2: Event) -> bool:
        """Check if two events have a temporal relationship"""
        # Check if both events have temporal expressions
        if not (event1.temporal_expressions and event2.temporal_expressions):
            return False
        
        # Check for temporal sequence indicators
        temporal_sequence_words = ['then', 'next', 'after', 'before', 'following', 'preceding']
        text2_lower = event2.description.lower()
        
        return any(word in text2_lower for word in temporal_sequence_words)

    def _has_entity_relationship(self, event1: Event, event2: Event) -> bool:
        """Check if two events share common entities"""
        # Extract potential entities (simple approach)
        entities1 = set(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', event1.description))
        entities2 = set(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', event2.description))
        
        # Check for overlap
        return len(entities1.intersection(entities2)) > 0
