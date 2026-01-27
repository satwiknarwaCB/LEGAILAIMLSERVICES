import re
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from dateutil import parser, relativedelta
import spacy
from app.models.chronology import TemporalExpression, Event

class TemporalProcessor:
    """Handles temporal expression extraction, normalization, and resolution"""
    
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to basic processing if spaCy model not available
            self.nlp = None
            print("Warning: spaCy model not found. Using basic temporal processing.")
        
        # Common temporal patterns
        self.temporal_patterns = {
            'absolute_dates': [
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ],
            'relative_dates': [
                r'\b(?:yesterday|today|tomorrow)\b',
                r'\b(?:last|next|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                r'\b\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|later|from now)\b',
                r'\b(?:in|after)\s+\d+\s+(?:days?|weeks?|months?|years?)\b',
                r'\b(?:before|after)\s+(?:the|this|next|last)\b'
            ],
            'time_expressions': [
                r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
                r'\b(?:morning|afternoon|evening|night|noon|midnight)\b'
            ]
        }
        
        # Relative date mappings
        self.relative_mappings = {
            'yesterday': -1,
            'today': 0,
            'tomorrow': 1,
            'last week': -7,
            'next week': 7,
            'this week': 0,
            'last month': -30,
            'next month': 30,
            'this month': 0,
            'last year': -365,
            'next year': 365,
            'this year': 0
        }

    def extract_temporal_expressions(self, text: str) -> List[TemporalExpression]:
        """Extract all temporal expressions from text"""
        temporal_expressions = []
        position = 0
        
        # Combine all patterns
        all_patterns = []
        for pattern_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                all_patterns.append((pattern, pattern_type))
        
        # Find all matches
        for pattern, pattern_type in all_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                expression = TemporalExpression(
                    text=match.group(),
                    position=match.start(),
                    is_relative=pattern_type == 'relative_dates'
                )
                temporal_expressions.append(expression)
        
        # Remove duplicates and sort by position
        unique_expressions = []
        seen_positions = set()
        for expr in sorted(temporal_expressions, key=lambda x: x.position):
            if expr.position not in seen_positions:
                unique_expressions.append(expr)
                seen_positions.add(expr.position)
        
        return unique_expressions

    def normalize_dates(self, temporal_expressions: List[TemporalExpression], 
                       reference_date: Optional[date] = None) -> List[TemporalExpression]:
        """Normalize temporal expressions to standard date format"""
        if reference_date is None:
            reference_date = date.today()
        
        normalized_expressions = []
        
        for expr in temporal_expressions:
            try:
                if expr.is_relative:
                    normalized_date = self._resolve_relative_date(expr.text, reference_date)
                else:
                    normalized_date = self._parse_absolute_date(expr.text)
                
                expr.normalized_date = normalized_date
                expr.confidence = self._calculate_confidence(expr.text, normalized_date)
                
            except Exception as e:
                print(f"Error normalizing date '{expr.text}': {e}")
                expr.confidence = 0.0
            
            normalized_expressions.append(expr)
        
        return normalized_expressions

    def _resolve_relative_date(self, date_text: str, reference_date: date) -> date:
        """Resolve relative date expressions to absolute dates"""
        date_text = date_text.lower().strip()
        
        # Handle simple relative dates
        if date_text in self.relative_mappings:
            days_offset = self.relative_mappings[date_text]
            return reference_date + timedelta(days=days_offset)
        
        # Handle "X days ago/later" patterns
        days_match = re.search(r'(\d+)\s+days?\s+(ago|later|from now)', date_text)
        if days_match:
            days = int(days_match.group(1))
            direction = days_match.group(2)
            if direction in ['ago']:
                return reference_date - timedelta(days=days)
            else:  # later, from now
                return reference_date + timedelta(days=days)
        
        # Handle "X weeks ago/later" patterns
        weeks_match = re.search(r'(\d+)\s+weeks?\s+(ago|later|from now)', date_text)
        if weeks_match:
            weeks = int(weeks_match.group(1))
            direction = weeks_match.group(2)
            if direction in ['ago']:
                return reference_date - timedelta(weeks=weeks)
            else:
                return reference_date + timedelta(weeks=weeks)
        
        # Handle "X months ago/later" patterns
        months_match = re.search(r'(\d+)\s+months?\s+(ago|later|from now)', date_text)
        if months_match:
            months = int(months_match.group(1))
            direction = months_match.group(2)
            if direction in ['ago']:
                return reference_date - relativedelta.relativedelta(months=months)
            else:
                return reference_date + relativedelta.relativedelta(months=months)
        
        # Handle "X years ago/later" patterns
        years_match = re.search(r'(\d+)\s+years?\s+(ago|later|from now)', date_text)
        if years_match:
            years = int(years_match.group(1))
            direction = years_match.group(2)
            if direction in ['ago']:
                return reference_date - relativedelta.relativedelta(years=years)
            else:
                return reference_date + relativedelta.relativedelta(years=years)
        
        # Default fallback
        return reference_date

    def _parse_absolute_date(self, date_text: str) -> date:
        """Parse absolute date expressions"""
        try:
            # Use dateutil parser for flexible date parsing
            parsed_date = parser.parse(date_text, fuzzy=True)
            return parsed_date.date()
        except Exception:
            # Fallback to basic parsing
            return self._basic_date_parse(date_text)

    def _basic_date_parse(self, date_text: str) -> date:
        """Basic date parsing fallback"""
        # Handle common formats
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_text, fmt).date()
            except ValueError:
                continue
        
        # If all parsing fails, return today's date
        return date.today()

    def _calculate_confidence(self, date_text: str, normalized_date: date) -> float:
        """Calculate confidence score for date normalization"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for well-formed dates
        if re.match(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', date_text):
            confidence += 0.3
        
        # Increase confidence for month names
        if any(month in date_text.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
            confidence += 0.2
        
        # Decrease confidence for relative dates (more ambiguous)
        if any(rel in date_text.lower() for rel in ['ago', 'later', 'yesterday', 'tomorrow']):
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))

    def resolve_relative_dates(self, events: List[Event], reference_date: date) -> List[Event]:
        """Resolve relative dates in events using a reference date"""
        resolved_events = []
        
        for event in events:
            # Process temporal expressions in the event
            for temp_expr in event.temporal_expressions:
                if temp_expr.is_relative and temp_expr.normalized_date is None:
                    try:
                        temp_expr.normalized_date = self._resolve_relative_date(
                            temp_expr.text, reference_date
                        )
                        temp_expr.confidence = self._calculate_confidence(
                            temp_expr.text, temp_expr.normalized_date
                        )
                    except Exception as e:
                        print(f"Error resolving relative date '{temp_expr.text}': {e}")
            
            # Update event's normalized date if it has temporal expressions
            if event.temporal_expressions:
                # Use the most confident temporal expression
                best_expr = max(event.temporal_expressions, 
                              key=lambda x: x.confidence, 
                              default=None)
                if best_expr and best_expr.normalized_date:
                    event.normalized_date = best_expr.normalized_date
            
            resolved_events.append(event)
        
        return resolved_events

    def validate_timeline_consistency(self, events: List[Event]) -> List[Dict[str, any]]:
        """Validate timeline consistency and detect conflicts"""
        conflicts = []
        
        # Sort events by normalized date
        sorted_events = sorted([e for e in events if e.normalized_date], 
                             key=lambda x: x.normalized_date)
        
        # Check for duplicate dates with different events
        date_groups = {}
        for event in sorted_events:
            if event.normalized_date not in date_groups:
                date_groups[event.normalized_date] = []
            date_groups[event.normalized_date].append(event)
        
        # Report conflicts
        for date_key, event_list in date_groups.items():
            if len(event_list) > 1:
                conflicts.append({
                    'type': 'duplicate_date',
                    'date': date_key,
                    'events': [{'id': e.id, 'description': e.description} for e in event_list],
                    'severity': 'medium'
                })
        
        # Check for impossible date sequences (future events before past events)
        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            # Check if events have conflicting temporal indicators
            current_has_future = any('later' in expr.text.lower() or 'next' in expr.text.lower() 
                                   for expr in current_event.temporal_expressions)
            next_has_past = any('ago' in expr.text.lower() or 'last' in expr.text.lower() 
                              for expr in next_event.temporal_expressions)
            
            if current_has_future and next_has_past:
                conflicts.append({
                    'type': 'temporal_sequence_conflict',
                    'events': [
                        {'id': current_event.id, 'description': current_event.description},
                        {'id': next_event.id, 'description': next_event.description}
                    ],
                    'severity': 'high'
                })
        
        return conflicts
