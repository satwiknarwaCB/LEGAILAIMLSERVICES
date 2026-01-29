# agent_registry.py - Registry of all available agents

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum


class AgentCapability(str, Enum):
    LEGAL_QA = "Legal Q&A"
    CASE_ANALYSIS = "Case Analysis"
    STATUTE_LOOKUP = "Statute Lookup"
    GENERAL_ADVICE = "General Advice"
    CONTRACT_DRAFTING = "Contract Drafting"
    AGREEMENT_TEMPLATES = "Agreement Templates"
    LEGAL_LETTERS = "Legal Letters"
    DOCUMENT_ANALYSIS = "Document Analysis"
    RISK_ASSESSMENT = "Risk Assessment"
    SUMMARY_GENERATION = "Summary Generation"
    CASE_LAW_SEARCH = "Case Law Search"
    PRECEDENT_ANALYSIS = "Precedent Analysis"
    CITATION_LOOKUP = "Citation Lookup"
    COMPLIANCE_CHECK = "Compliance Check"
    REGULATION_LOOKUP = "Regulation Lookup"
    VIOLATION_DETECTION = "Violation Detection"
    CONTRACT_REVIEW = "Contract Review"
    RISK_HIGHLIGHTING = "Risk Highlighting"
    TERM_EXTRACTION = "Term Extraction"


class AgentDefinition(BaseModel):
    id: str
    name: str
    description: str
    capabilities: List[str]
    is_active: bool = True
    requires_document: bool = False
    backend_service: str  # Which service handles this agent
    backend_port: int
    backend_endpoint: Optional[str] = None  # Fixed: changed to Optional[str]


# Define all available agents
AGENT_REGISTRY: Dict[str, AgentDefinition] = {
    "ask": AgentDefinition(
        id="ask",
        name="Ask Agent",
        description="Get answers to any legal question with AI-powered insights",
        capabilities=["Legal Q&A", "Case Analysis", "Statute Lookup", "General Advice"],
        is_active=True,
        requires_document=False,
        backend_service="ASK_DRAFT",
        backend_port=8000,
        backend_endpoint="/api/chat"
    ),
    "draft": AgentDefinition(
        id="draft",
        name="Draft Agent",
        description="Generate professional legal documents and contracts",
        capabilities=["Contract Drafting", "Agreement Templates", "Legal Letters"],
        is_active=True,
        requires_document=False,
        backend_service="ASK_DRAFT",
        backend_port=8000,
        backend_endpoint="/api/drafting/generate"
    ),
    "interact": AgentDefinition(
        id="interact",
        name="Interact Agent",
        description="Analyze and interact with your uploaded legal documents",
        capabilities=["Document Analysis", "Risk Assessment", "Summary Generation"],
        is_active=True,
        requires_document=True,
        backend_service="INTERACT",
        backend_port=8001,
        backend_endpoint="/analyze"
    ),
    "research": AgentDefinition(
        id="research",
        name="Research Agent",
        description="Deep dive into case law and legal precedents",
        capabilities=["Case Law Search", "Precedent Analysis", "Citation Lookup"],
        is_active=True,
        requires_document=False,
        backend_service="AGENTS_DIRECT",
        backend_port=8002,
        backend_endpoint=None  # Handled directly by agents service
    ),
    "compliance": AgentDefinition(
        id="compliance",
        name="Compliance Agent",
        description="Check regulatory compliance and identify violations",
        capabilities=["Compliance Check", "Regulation Lookup", "Violation Detection"],
        is_active=True,
        requires_document=False,
        backend_service="AGENTS_DIRECT",
        backend_port=8002,
        backend_endpoint=None
    ),
    "review": AgentDefinition(
        id="review",
        name="Review Agent",
        description="Review contracts and highlight key terms and risks",
        capabilities=["Contract Review", "Risk Highlighting", "Term Extraction"],
        is_active=True,
        requires_document=True,
        backend_service="AGENTS_DIRECT",
        backend_port=8002,
        backend_endpoint=None
    )
}


def get_all_agents() -> List[Dict[str, Any]]:
    """Get list of all agents for API response"""
    return [
        {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "is_active": agent.is_active,
            "requires_document": agent.requires_document
        }
        for agent in AGENT_REGISTRY.values()
    ]


def get_agent(agent_id: str) -> AgentDefinition:
    """Get a specific agent by ID"""
    return AGENT_REGISTRY.get(agent_id)


def is_valid_agent(agent_id: str) -> bool:
    """Check if agent ID is valid"""
    return agent_id in AGENT_REGISTRY
