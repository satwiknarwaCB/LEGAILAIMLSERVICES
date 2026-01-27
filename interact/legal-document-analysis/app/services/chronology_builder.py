import uuid
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from app.models.chronology import Event, Timeline, EventType
from app.services.temporal_processor import TemporalProcessor
from app.services.event_extractor import EventExtractor

class ChronologyBuilder:
    """Builds and manages chronological timelines from extracted events"""
    
    def __init__(self):
        self.temporal_processor = TemporalProcessor()
        self.event_extractor = EventExtractor()

    def build_timeline(self, events: List[Event], document_id: str, 
                      reference_date: Optional[date] = None) -> Timeline:
        """Build a complete timeline from events"""
        if reference_date is None:
            reference_date = date.today()
        
        # Resolve relative dates
        resolved_events = self.temporal_processor.resolve_relative_dates(events, reference_date)
        
        # Order events chronologically
        ordered_events = self.order_events_chronologically(resolved_events)
        
        # Calculate timeline statistics
        date_range = self._calculate_date_range(ordered_events)
        confidence_summary = self._calculate_confidence_summary(ordered_events)
        
        # Detect temporal conflicts
        temporal_conflicts = self.temporal_processor.validate_timeline_consistency(ordered_events)
        
        # Create timeline
        timeline = Timeline(
            document_id=document_id,
            events=ordered_events,
            total_events=len(ordered_events),
            date_range=date_range,
            confidence_summary=confidence_summary,
            temporal_conflicts=temporal_conflicts
        )
        
        return timeline

    def order_events_chronologically(self, events: List[Event]) -> List[Event]:
        """Order events chronologically based on their temporal information"""
        # Separate events with and without dates
        events_with_dates = []
        events_without_dates = []
        
        for event in events:
            if event.normalized_date:
                events_with_dates.append(event)
            else:
                events_without_dates.append(event)
        
        # Sort events with dates chronologically
        events_with_dates.sort(key=lambda x: x.normalized_date)
        
        # For events without dates, try to infer order from context
        ordered_events_without_dates = self._infer_event_order(events_without_dates)
        
        # Combine ordered events
        all_ordered_events = events_with_dates + ordered_events_without_dates
        
        # Add sequence numbers
        for i, event in enumerate(all_ordered_events):
            event.metadata['sequence_number'] = i + 1
        
        return all_ordered_events

    def detect_temporal_conflicts(self, timeline: Timeline) -> List[Dict[str, Any]]:
        """Detect and analyze temporal conflicts in the timeline"""
        conflicts = []
        
        # Check for duplicate dates
        date_groups = {}
        for event in timeline.events:
            if event.normalized_date:
                if event.normalized_date not in date_groups:
                    date_groups[event.normalized_date] = []
                date_groups[event.normalized_date].append(event)
        
        # Report conflicts
        for date_key, event_list in date_groups.items():
            if len(event_list) > 1:
                conflicts.append({
                    'type': 'duplicate_date',
                    'date': date_key.isoformat(),
                    'events': [{'id': e.id, 'description': e.description} for e in event_list],
                    'severity': 'medium',
                    'suggestion': 'Consider if these events occurred at different times of the same day'
                })
        
        # Check for impossible sequences
        for i in range(len(timeline.events) - 1):
            current_event = timeline.events[i]
            next_event = timeline.events[i + 1]
            
            if (current_event.normalized_date and next_event.normalized_date and
                current_event.normalized_date > next_event.normalized_date):
                conflicts.append({
                    'type': 'chronological_sequence_error',
                    'events': [
                        {'id': current_event.id, 'description': current_event.description, 'date': current_event.normalized_date.isoformat()},
                        {'id': next_event.id, 'description': next_event.description, 'date': next_event.normalized_date.isoformat()}
                    ],
                    'severity': 'high',
                    'suggestion': 'Review the chronological order of these events'
                })
        
        return conflicts

    def generate_chronology_report(self, timeline: Timeline) -> Dict[str, Any]:
        """Generate a comprehensive chronology report"""
        report = {
            'summary': {
                'total_events': timeline.total_events,
                'date_range': timeline.date_range,
                'confidence_avg': timeline.confidence_summary.get('average', 0.0),
                'temporal_conflicts': len(timeline.temporal_conflicts)
            },
            'events_by_type': self._group_events_by_type(timeline.events),
            'events_by_date': self._group_events_by_date(timeline.events),
            'confidence_analysis': timeline.confidence_summary,
            'temporal_conflicts': timeline.temporal_conflicts,
            'recommendations': self._generate_recommendations(timeline)
        }
        
        return report

    def export_timeline(self, timeline: Timeline, format: str = 'json') -> Dict[str, Any]:
        """Export timeline in various formats"""
        if format.lower() == 'json':
            return self._export_json(timeline)
        elif format.lower() == 'csv':
            return self._export_csv(timeline)
        elif format.lower() == 'summary':
            return self._export_summary(timeline)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _calculate_date_range(self, events: List[Event]) -> Dict[str, Optional[date]]:
        """Calculate the date range of events"""
        dates = [event.normalized_date for event in events if event.normalized_date]
        
        if not dates:
            return {'start': None, 'end': None}
        
        return {
            'start': min(dates),
            'end': max(dates)
        }

    def _calculate_confidence_summary(self, events: List[Event]) -> Dict[str, float]:
        """Calculate confidence statistics for events"""
        if not events:
            return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        
        confidences = [event.confidence_score for event in events]
        
        return {
            'average': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'std': self._calculate_std_dev(confidences)
        }

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _infer_event_order(self, events: List[Event]) -> List[Event]:
        """Infer order for events without explicit dates"""
        # Simple heuristic: order by position in document
        return sorted(events, key=lambda x: x.metadata.get('sentence_index', 0))

    def _group_events_by_type(self, events: List[Event]) -> Dict[str, List[Dict[str, Any]]]:
        """Group events by their type"""
        groups = {}
        
        for event in events:
            event_type = event.event_type.value
            if event_type not in groups:
                groups[event_type] = []
            
            groups[event_type].append({
                'id': event.id,
                'description': event.description,
                'date': event.normalized_date.isoformat() if event.normalized_date else None,
                'confidence': event.confidence_score
            })
        
        return groups

    def _group_events_by_date(self, events: List[Event]) -> Dict[str, List[Dict[str, Any]]]:
        """Group events by their date"""
        groups = {}
        
        for event in events:
            if event.normalized_date:
                date_key = event.normalized_date.isoformat()
                if date_key not in groups:
                    groups[date_key] = []
                
                groups[date_key].append({
                    'id': event.id,
                    'description': event.description,
                    'type': event.event_type.value,
                    'confidence': event.confidence_score
                })
        
        return groups

    def _generate_recommendations(self, timeline: Timeline) -> List[str]:
        """Generate recommendations for improving the timeline"""
        recommendations = []
        
        # Check confidence levels
        if timeline.confidence_summary.get('average', 0) < 0.5:
            recommendations.append("Consider manual review of low-confidence events")
        
        # Check for conflicts
        if timeline.temporal_conflicts:
            recommendations.append(f"Resolve {len(timeline.temporal_conflicts)} temporal conflicts")
        
        # Check for events without dates
        events_without_dates = [e for e in timeline.events if not e.normalized_date]
        if events_without_dates:
            recommendations.append(f"Review {len(events_without_dates)} events without explicit dates")
        
        # Check date range
        if timeline.date_range['start'] and timeline.date_range['end']:
            date_span = (timeline.date_range['end'] - timeline.date_range['start']).days
            if date_span > 365:
                recommendations.append("Consider breaking down long timeline into smaller periods")
        
        return recommendations

    def _export_json(self, timeline: Timeline) -> Dict[str, Any]:
        """Export timeline as JSON"""
        return {
            'document_id': timeline.document_id,
            'created_at': timeline.created_at.isoformat(),
            'total_events': timeline.total_events,
            'date_range': {
                'start': timeline.date_range['start'].isoformat() if timeline.date_range['start'] else None,
                'end': timeline.date_range['end'].isoformat() if timeline.date_range['end'] else None
            },
            'events': [
                {
                    'id': event.id,
                    'description': event.description,
                    'type': event.event_type.value,
                    'date': event.normalized_date.isoformat() if event.normalized_date else None,
                    'confidence': event.confidence_score,
                    'source_text': event.source_text,
                    'document_section': event.document_section,
                    'metadata': event.metadata
                }
                for event in timeline.events
            ],
            'confidence_summary': timeline.confidence_summary,
            'temporal_conflicts': timeline.temporal_conflicts
        }

    def _export_csv(self, timeline: Timeline) -> Dict[str, Any]:
        """Export timeline as CSV data"""
        csv_data = []
        
        # Header
        csv_data.append([
            'Event ID', 'Description', 'Type', 'Date', 'Confidence', 
            'Source Text', 'Document Section', 'Sequence Number'
        ])
        
        # Data rows
        for event in timeline.events:
            csv_data.append([
                event.id,
                event.description,
                event.event_type.value,
                event.normalized_date.isoformat() if event.normalized_date else '',
                str(event.confidence_score),
                event.source_text,
                event.document_section,
                str(event.metadata.get('sequence_number', ''))
            ])
        
        return {
            'format': 'csv',
            'data': csv_data,
            'filename': f"chronology_{timeline.document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }

    def _export_summary(self, timeline: Timeline) -> Dict[str, Any]:
        """Export timeline as a summary report"""
        return {
            'format': 'summary',
            'document_id': timeline.document_id,
            'summary': {
                'total_events': timeline.total_events,
                'date_range': timeline.date_range,
                'average_confidence': timeline.confidence_summary.get('average', 0.0),
                'temporal_conflicts': len(timeline.temporal_conflicts)
            },
            'events_by_type': self._group_events_by_type(timeline.events),
            'recommendations': self._generate_recommendations(timeline)
        }
