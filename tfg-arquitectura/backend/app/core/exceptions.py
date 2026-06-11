import structlog
from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

log = structlog.get_logger()


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    log.warning("http_exception", path=str(request.url), status=exc.status_code, detail=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_exception", path=str(request.url))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
