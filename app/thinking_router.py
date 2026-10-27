import os
import re
import time
import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Nemotron-Nano-9B-v2 Adaptive Thinking Router")

VLLM_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("VLLM_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8")
DRY_RUN = os.getenv("ROUTER_DRY_RUN", "true").lower() == "true"

REASONING_PATTERN = re.compile(
    r"\b(calculate|derive|prove|step by step|math|equation|algorithm|optimize|"
    r"roofline|bandwidth|complexity|debug|why does|implement|leetcode|sql)\b|[0-9]+\s*[\+\-\*\/]\s*[0-9]+",
    re.IGNORECASE,
)

METRICS = {
    "reasoning_off_requests": 0,
    "reasoning_on_requests": 0,
    "estimated_tokens_saved": 0,
    "estimated_vram_block_seconds_saved": 0.0,
}

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 350
    force_mode: Optional[str] = None

def classify_prompt(prompt: str, force_mode: Optional[str] = None) -> tuple[str, str]:
    if force_mode == "think":
        return "REASONING_ON", "detailed thinking on"
    if force_mode == "no_think":
        return "REASONING_OFF", "/no_think"
    if REASONING_PATTERN.search(prompt) or len(prompt.split()) > 120:
        return "REASONING_ON", "detailed thinking on"
    return "REASONING_OFF", "/no_think"

@app.post("/v1/route")
async def route_request(req: ChatRequest):
    t0 = time.perf_counter()
    mode, sys_prompt = classify_prompt(req.prompt, req.force_mode)

    if mode == "REASONING_OFF":
        METRICS["reasoning_off_requests"] += 1
        METRICS["estimated_tokens_saved"] += 250
        METRICS["estimated_vram_block_seconds_saved"] += round(250 / 23.30, 2)
    else:
        METRICS["reasoning_on_requests"] += 1

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": req.prompt},
        ],
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
    }

    if DRY_RUN:
        return {
            "routed_mode": mode,
            "injected_system_prompt": sys_prompt,
            "target_model": MODEL_NAME,
            "vram_block_impact": (
                "Frees 1,312-token Mamba/KV block ~3.5x faster"
                if mode == "REASONING_OFF"
                else "Allocates full reasoning budget for multi-step accuracy"
            ),
            "router_overhead_ms": round((time.perf_counter() - t0) * 1000, 3),
            "upstream_payload": payload,
        }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(VLLM_URL, json=payload)
        data = resp.json()

    return {
        "routed_mode": mode,
        "injected_system_prompt": sys_prompt,
        "router_overhead_ms": round((time.perf_counter() - t0) * 1000, 3),
        "vllm_response": data,
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    lines = [
        f'nemotron_router_requests_total{{mode="reasoning_off"}} {METRICS["reasoning_off_requests"]}',
        f'nemotron_router_requests_total{{mode="reasoning_on"}} {METRICS["reasoning_on_requests"]}',
        f'nemotron_router_estimated_tokens_saved_total {METRICS["estimated_tokens_saved"]}',
        f'nemotron_router_vram_block_seconds_saved_total {METRICS["estimated_vram_block_seconds_saved"]:.2f}',
    ]
    return "\n".join(lines) + "\n"
