from fastapi import FastAPI , APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

router = APIRouter()



@router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=router.openapi_url,
        title=router.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@router.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(title=router.title, version=router.version, routes=router.routes)

