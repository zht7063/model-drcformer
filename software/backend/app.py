"""HTTP application. GPU work is serialized in a single dedicated thread."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
import hmac
import io
import json
import logging
import time
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from python_multipart.exceptions import MultipartParseError

from software.backend.config import Settings
from software.backend.storage import ResultStore
from software.backend.schemas import (ErrorResponse, HealthResponse, InferenceParams,
                                      ModelsResponse, SegmentResponse)

LOGGER = logging.getLogger(__name__)
STAGES = {"segmentation", "slic", "cam"}
FORM_FIELDS = {"image", "include", "model", "params"}


def decode_image(data, max_pixels):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "unsupported_image_format")
                if source.width * source.height > max_pixels:
                    raise HTTPException(413, "image_pixel_limit")
                if getattr(source, "n_frames", 1) != 1:
                    raise HTTPException(415, "animated_image_not_supported")
                return ImageOps.exif_transpose(source).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(413, "image_pixel_limit")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "invalid_image")


class AdmissionMiddleware:
    """Authenticate and bound uploads + queued + active work before body parsing."""
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if scope["path"] not in {"/health/live", "/health/ready"}:
            expected = ("Bearer " + self.settings.api_key).encode("ascii")
            if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                return await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
        admitted = scope["method"] == "POST" and scope["path"].rstrip("/") == "/api/v1/segment"
        if not admitted:
            return await self.app(scope, receive, send)
        if self.active >= self.settings.queue_capacity + 1:
            return await JSONResponse({"error": "server_busy"}, status_code=429,
                                      headers={"Retry-After": "2"})(scope, receive, send)
        self.active += 1
        try:
            return await self.app(scope, receive, send)
        finally:
            self.active -= 1


def create_app(settings=None, engine_factory=None):
    settings = settings or Settings.from_env()
    settings.validate()
    store = ResultStore(settings.results_dir, settings.result_ttl_seconds)

    if engine_factory is None:
        def engine_factory():
            from software.backend.engine import SegmentationEngine
            return SegmentationEngine(settings.models, settings.device, settings.max_inference_side,
                                      gt_dir=settings.gt_dir)

    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmentation")
        app.state.executor = executor
        app.state.ready = False
        cleanup_task = None
        try:
            app.state.engine = await asyncio.get_running_loop().run_in_executor(executor, engine_factory)
            app.state.ready = True

            async def sweep():
                while True:
                    await asyncio.to_thread(store.cleanup)
                    await asyncio.sleep(60)

            cleanup_task = asyncio.create_task(sweep())
            yield
        finally:
            app.state.ready = False
            if cleanup_task:
                cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup_task
            executor.shutdown(wait=True, cancel_futures=True)
            app.state.engine = None

    app = FastAPI(title="DRCFormer Segmentation Service", version="0.1.0", lifespan=lifespan,
                  responses={code: {"model": ErrorResponse} for code in
                             (400, 401, 404, 408, 413, 415, 429, 500, 503)})
    app.state.ready = False
    app.add_middleware(AdmissionMiddleware, settings=settings)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(MultipartParseError)
    async def malformed_multipart(request, exc):
        return JSONResponse({"error": "invalid_multipart"}, status_code=400)

    @app.get("/health/live", response_model=HealthResponse)
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready", response_model=HealthResponse,
             responses={503: {"model": HealthResponse}})
    async def ready():
        return JSONResponse({"status": "ready" if app.state.ready else "not_ready"},
                            status_code=200 if app.state.ready else 503)

    @app.get("/api/v1/models", response_model=ModelsResponse)
    async def models():
        if not app.state.ready:
            raise HTTPException(503, "model_not_ready")
        return {"default_model": settings.models[0].id,
                "models": app.state.engine.metadata(), "limits": {
            name: getattr(settings, name) for name in
            ("max_upload_bytes", "max_pixels", "max_inference_side", "result_ttl_seconds",
             "upload_timeout_seconds", "queue_capacity")}}

    def process(data, include, model_id, params, filename=None):
        started = time.perf_counter()
        image = decode_image(data, settings.max_pixels)
        request_id, directory = store.create()
        try:
            result = app.state.engine.run(image, include, directory, model_id, params,
                                          filename=filename)
            result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            result = store.publish(request_id, result)
            LOGGER.info("request=%s duration_ms=%s stages=%s", request_id,
                        result["duration_ms"], {k: v["status"] for k, v in result["stages"].items()})
            return result
        except Exception:
            store.discard(request_id)
            LOGGER.exception("request=%s inference failed", request_id)
            raise

    @app.post("/api/v1/segment", response_model=SegmentResponse,
              response_model_exclude_none=True, openapi_extra={"requestBody": {
        "required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["image"], "properties": {
                "image": {"type": "string", "format": "binary"},
                "include": {"type": "string", "default": '["segmentation"]'},
                "model": {"type": "string", "description": "model id from /api/v1/models"},
                "params": {"type": "string", "default": "{}",
                           "description": "JSON object of InferenceParams fields"},
            }}}}}})
    async def segment(request: Request):
        if not app.state.ready:
            raise HTTPException(503, "model_not_ready")
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
            raise HTTPException(415, "multipart_form_required")
        # Bound raw bytes even for chunked requests without Content-Length, before multipart parsing.
        async def read_body():
            chunks, total = [], 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > settings.max_upload_bytes + 65536:
                    raise HTTPException(413, "upload_limit")
                chunks.append(chunk)
            return b"".join(chunks)

        try:
            body = await asyncio.wait_for(read_body(), settings.upload_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(408, "upload_timeout")
        delivered = False

        async def body_receive():
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        parsed_request = Request(request.scope, body_receive)
        async with parsed_request.form(max_files=1, max_fields=3, max_part_size=4096) as form:
            if (any(key not in FORM_FIELDS for key in form)
                    or any(len(form.getlist(key)) > 1 for key in FORM_FIELDS)):
                raise HTTPException(400, "unknown_or_duplicate_form_field")
            upload = form.get("image")
            if not isinstance(upload, UploadFile):
                raise HTTPException(400, "image_file_required")
            filename = upload.filename
            data = await upload.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "upload_limit")
            try:
                include = json.loads(form.get("include", '["segmentation"]'))
            except (TypeError, ValueError):
                raise HTTPException(400, "invalid_include")
            if (not isinstance(include, list) or any(not isinstance(x, str) for x in include)
                    or not set(include) <= STAGES):
                raise HTTPException(400, "invalid_include")
            model_id = form.get("model") or settings.models[0].id
            if model_id not in {spec.id for spec in settings.models}:
                raise HTTPException(400, "unknown_model")
            try:
                params = InferenceParams.model_validate_json(form.get("params") or "{}")
            except (TypeError, ValidationError):
                raise HTTPException(400, "invalid_params")
        future = asyncio.get_running_loop().run_in_executor(
            app.state.executor, process, data, set(include), model_id, params, filename)
        try:
            # A disconnected caller must not release capacity while its GPU job still runs.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            with suppress(Exception):
                await asyncio.shield(future)
            raise
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(500, "inference_failed")

    @app.get("/api/v1/results/{request_id}/{artifact}")
    async def artifact(request_id: str, artifact: str):
        try:
            content = await asyncio.to_thread(store.read, request_id, artifact)
        except (OSError, KeyError, ValueError):
            raise HTTPException(404, "result_not_found_or_expired")
        media = "image/png" if artifact.endswith(".png") else (
            "application/json" if artifact.endswith(".json") else "application/octet-stream")
        return Response(content, media_type=media, headers={"Cache-Control": "no-store",
                        "Content-Disposition": f'attachment; filename="{artifact}"'})

    def openapi():
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
            schema.setdefault("components", {})["securitySchemes"] = {
                "BearerAuth": {"type": "http", "scheme": "bearer"}}
            for path, operations in schema["paths"].items():
                if path.startswith("/api/"):
                    for operation in operations.values():
                        operation["security"] = [{"BearerAuth": []}]
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi

    return app
