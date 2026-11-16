"""HTTP API for template inspection and generation."""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dslm.query import ModelBundle, load_bundle
from dslm.template import sample_training_text


class QueryBody(BaseModel):
    question: str | None = None
    system: str | None = None
    messages: list[dict[str, str]] | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)


def create_app(cfg: dict[str, Any], run_dir: str | None) -> FastAPI:
    app = FastAPI(title="DSLM", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state: dict[str, Any] = {
        "cfg": cfg,
        "run_dir": run_dir,
        "bundle": None,
        "model_status": "idle",
        "error": None,
    }
    lock = threading.Lock()

    def ensure_loaded() -> ModelBundle:
        with lock:
            if state["bundle"] is not None:
                return state["bundle"]
            if state["model_status"] == "loading":
                raise HTTPException(status_code=409, detail="The model is still loading.")
            state["model_status"] = "loading"
            state["error"] = None
        try:
            bundle = load_bundle(cfg, run_dir)
        except Exception as exc:  # surface download and weight errors to the client
            with lock:
                state["model_status"] = "error"
                state["error"] = str(exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        with lock:
            state["bundle"] = bundle
            state["model_status"] = "ready"
        return bundle

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        bundle: ModelBundle | None = state["bundle"]
        return {
            "status": "ok",
            "model_status": state["model_status"],
            "model_name": cfg["model"]["name"],
            "run_dir": run_dir,
            "using_adapter": bool(bundle.using_adapter) if bundle else bool(run_dir),
            "error": state["error"],
        }

    @app.get("/api/template")
    def template() -> dict[str, Any]:
        bundle: ModelBundle | None = state["bundle"]
        sample = sample_training_text(cfg["model"]["name"], bundle.tokenizer if bundle else None)
        return sample

    @app.get("/api/examples")
    def examples() -> dict[str, Any]:
        return {
            "prompts": [
                {
                    "label": "Pune cutoff",
                    "question": "What is the cutoff for same-day dispatch from the Pune hub?",
                },
                {
                    "label": "Damage claims",
                    "question": "How long do I have to file a damage claim?",
                },
                {
                    "label": "Drone delivery",
                    "question": "Does Harborline offer drone delivery?",
                },
                {
                    "label": "COD remittance",
                    "question": "When is COD cash remitted to the shipper?",
                },
            ]
        }

    @app.post("/api/query")
    def query(body: QueryBody) -> dict[str, Any]:
        bundle = ensure_loaded()
        if body.max_new_tokens is not None:
            bundle.max_new_tokens = body.max_new_tokens
        try:
            messages = bundle.prepare_messages(body.messages, body.question, body.system)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = bundle.generate(messages)
        result["model_name"] = bundle.model_name
        result["using_adapter"] = bundle.using_adapter
        result["run_dir"] = str(bundle.run_dir) if bundle.run_dir else None
        return result

    @app.post("/api/load")
    def load() -> dict[str, Any]:
        ensure_loaded()
        return health()

    return app
