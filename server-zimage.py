"""
Z-Image-Turbo HTTP server for RunPod Pods (alternative to ERNIE).

Same API surface as server.py but with Tongyi-MAI/Z-Image-Turbo loaded
via ZImagePipeline. Runs ~3-4× faster than ERNIE on the same A40 (4 steps
vs 8, 6B params vs 8B).

Run with: python server-zimage.py

Env vars:
  MODEL_ID        default "Tongyi-MAI/Z-Image-Turbo"
  HF_TOKEN        optional
  POD_API_KEY     optional Bearer auth
  PORT            default 8000
"""
import base64
import io
import os
import random
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from PIL import Image
import uvicorn

# ─── Config ───
MODEL_ID = os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
API_KEY = os.environ.get("POD_API_KEY") or None
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))

# Z-Image trained on ~1024-class. These ratios stay close to that area.
RATIOS = {
    "1:1":  (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1152, 896),
    "3:4":  (896, 1152),
    "3:2":  (1216, 832),
    "2:3":  (832, 1216),
}

state: dict = {"pipe": None, "load_error": None, "load_ms": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[server-zimage] Python {sys.version.split()[0]}", flush=True)
    print(f"[server-zimage] PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[server-zimage] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)", flush=True)

    try:
        import diffusers
        print(f"[server-zimage] diffusers {diffusers.__version__}", flush=True)
        from diffusers import ZImagePipeline
    except Exception as e:
        state["load_error"] = f"diffusers import: {type(e).__name__}: {e}"
        print(f"[server-zimage] ✕ FATAL: {state['load_error']}", flush=True)
        print(traceback.format_exc(), flush=True)
        yield
        return

    print(f"[server-zimage] Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    try:
        kwargs = {"torch_dtype": torch.bfloat16}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        pipe = ZImagePipeline.from_pretrained(MODEL_ID, **kwargs).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        # NO attention_slicing on A40 — slows inference 20-30% with no benefit
        state["pipe"] = pipe
        state["load_ms"] = int((time.time() - t0) * 1000)
        print(f"[server-zimage] ✓ Model ready in {state['load_ms']/1000:.1f}s — ready to serve", flush=True)
    except Exception as e:
        state["load_error"] = f"{type(e).__name__}: {e}"
        print(f"[server-zimage] ✕ Model load failed: {state['load_error']}", flush=True)
        print(traceback.format_exc(), flush=True)
    yield


app = FastAPI(title="Z-Image-Turbo Pod Server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    ratio: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: int = 4              # Z-Image is 4-step distilled
    guidance_scale: float = 1.0  # Z-Image doesn't use CFG (>1.0 hurts)
    seed: Optional[int] = None
    output_format: str = "jpg"
    # Note: NO use_pe — Z-Image doesn't have a built-in prompt enhancer.


@app.get("/health")
def health():
    return {
        "ready": state["pipe"] is not None,
        "model_id": MODEL_ID,
        "load_ms": state["load_ms"],
        "load_error": state["load_error"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pipeline": "ZImagePipeline",
    }


def _check_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <key>")
    if authorization.split(" ", 1)[1] != API_KEY:
        raise HTTPException(401, "Invalid token")


@app.post("/generate")
def generate(req: GenerateRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    if state["pipe"] is None:
        raise HTTPException(503, f"Model not ready: {state['load_error'] or 'still loading'}")

    if req.ratio and req.ratio in RATIOS:
        w, h = RATIOS[req.ratio]
    elif req.width and req.height:
        w, h = int(req.width), int(req.height)
    else:
        w, h = RATIOS["16:9"]
    w = max(64, round(w / 64) * 64)
    h = max(64, round(h / 64) * 64)

    seed = req.seed if (req.seed is not None and req.seed >= 0) else random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    try:
        t0 = time.time()
        with torch.inference_mode():
            result = state["pipe"](
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=w,
                height=h,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
                # NO use_pe arg — Z-Image rejects unknown kwargs
            )
        img: Image.Image = result.images[0]
        gen_ms = int((time.time() - t0) * 1000)

        buf = io.BytesIO()
        fmt = "JPEG" if req.output_format.lower() in ("jpg", "jpeg") else "PNG"
        if fmt == "JPEG":
            img = img.convert("RGB")
            img.save(buf, format=fmt, quality=92, optimize=True)
        else:
            img.save(buf, format=fmt, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "image_base64": b64,
            "format": fmt.lower(),
            "width": w,
            "height": h,
            "seed": int(seed),
            "steps": req.steps,
            "guidance_scale": req.guidance_scale,
            "use_pe": False,  # always false for Z-Image (kept for client compat)
            "generation_ms": gen_ms,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise HTTPException(507, f"CUDA OOM: {e}. Try smaller resolution.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[server-zimage] generate ERROR:\n{tb}", flush=True)
        raise HTTPException(500, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    print(f"[server-zimage] Starting on {HOST}:{PORT}…", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
