from fastapi import APIRouter

from koshshield.api.routes import agent, audit, documents, health, retrieval, review

api_router = APIRouter()
api_router.include_router(health.router, tags=["system"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(audit.router, prefix="/audit", tags=["audit"])
api_router.include_router(review.router, tags=["review"])
api_router.include_router(retrieval.router, tags=["retrieval"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
