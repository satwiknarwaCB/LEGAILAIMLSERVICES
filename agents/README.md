# Agents Service - Legal AI Agents Gateway

This service provides a unified API for all legal AI agents.

## Port: 8002

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents` | GET | List all available agents |
| `/api/agents/{agent_id}/session` | POST | Create a new session |
| `/api/agents/{agent_id}/chat` | POST | Chat with an agent |
| `/api/agents/{agent_id}/session/{session_id}/history` | GET | Get chat history |
| `/api/agents/{agent_id}/upload` | POST | Upload document (interact/review) |
| `/api/agents/{agent_id}/session/{session_id}` | DELETE | End session |

## Agents

| Agent ID | Name | Description |
|----------|------|-------------|
| `ask` | Ask Agent | Legal Q&A (routes to port 8000) |
| `draft` | Draft Agent | Document drafting (routes to port 8000) |
| `interact` | Interact Agent | Document analysis (routes to port 8001) |
| `research` | Research Agent | Case law research (direct LLM) |
| `compliance` | Compliance Agent | Regulatory compliance check (direct LLM) |
| `review` | Review Agent | Contract review & risk analysis (direct LLM) |

## Run

```bash
cd agents
python -m uvicorn agents_api:app --host 0.0.0.0 --port 8002 --reload
```

## Structure

```
agents/
├── agents_api.py           # Main FastAPI app
├── agent_registry.py       # Agent definitions
├── session_manager.py      # Session management
├── requirements.txt        # Dependencies
└── agent_handlers/         # Individual agent implementations
    ├── __init__.py
    ├── base_agent.py       # Base class
    ├── ask_agent.py        # Routes to chatbot
    ├── draft_agent.py      # Routes to drafting
    ├── interact_agent.py   # Document analysis
    ├── research_agent.py   # Legal research
    ├── compliance_agent.py # Compliance check
    └── review_agent.py     # Contract review
```
