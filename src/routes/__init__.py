from fastapi import APIRouter

from src.routes.llm_agents_router import router as llm_agents_router

router = APIRouter()
router.include_router(llm_agents_router)
