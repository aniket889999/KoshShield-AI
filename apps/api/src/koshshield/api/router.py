from fastapi import APIRouter

from koshshield.api.routes import audit, documents, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["system"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(audit.router, prefix="/audit", tags=["audit"])
