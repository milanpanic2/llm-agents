from fastapi import APIRouter

from src.routes.llm_agents_router import router as web_search_router

router = APIRouter()
router.include_router(web_search_router)
