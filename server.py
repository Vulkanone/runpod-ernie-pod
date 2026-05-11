"""
ERNIE-Image-Turbo HTTP server for RunPod Pods.

Runs as a regular long-lived Python process inside a Pod (not Serverless).
Exposes /generate over HTTP. No cold start after initial load.

Endpoints:
  GET  /health         → liveness + readiness probe
  POST /generate       → generate one image (JSON in, JSON out)

Env vars:
  MODEL_ID        default "Baidu/ERNIE-Image-Turbo"
  HF_TOKEN        optional, for private/gated models
  POD_API_KEY     optional; if set, /generate requires `Authorization: Bearer <key>`
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
MODEL_ID = os.environ.get("MODEL_ID", "Baidu/ERNIE-Image-Turbo")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
API_KEY = os.environ.get("POD_API_KEY") or None  # None = no auth (URL is the secret)
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))

# Baidu's recommended ratios (close to 1 MP each)
RATIOS = {
    "1:1":  (1024, 1024),
    "16:9": (1376, 768),
    "9:16": (768, 1376),
    "4:3":  (1200, 896),
    "3:4":  (896, 1200),
    "3:2":  (1264, 848),
    "2:3":  (848, 1264),
}

# ─── State (set at startup, used by /generate) ───
state: dict = {"pipe": None, "load_error": None, "load_ms": 0}


# ─── Model load on FastAPI startup ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[server] Python {sys.version.split()[0]}", flush=True)
    print(f"[server] PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[server] GPU: {gpu_name} ({gpu_vram:.1f} GB)", flush=True)

    try:
        import diffusers
        print(f"[server] diffusers {diffusers.__version__}", flush=True)
        from diffusers import ErnieImagePipeline
    except Exception as e:
        state["load_error"] = f"diffusers import: {type(e).__name__}: {e}"
        print(f"[server] ✕ FATAL: {state['load_error']}", flush=True)
        print(traceback.format_exc(), flush=True)
        yield
        return

    print(f"[server] Loading {MODEL_ID}…", flush=True)
    t0 = time.time()
    try:
        kwargs = {"torch_dtype": torch.bfloat16}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        pipe = ErnieImagePipeline.from_pretrained(MODEL_ID, **kwargs).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        state["pipe"] = pipe
        state["load_ms"] = int((time.time() - t0) * 1000)
        print(f"[server] ✓ Model ready in {state['load_ms']/1000:.1f}s — ready to serve", flush=True)
    except Exception as e:
        state["load_error"] = f"{type(e).__name__}: {e}"
        print(f"[server] ✕ Model load failed: {state['load_error']}", flush=True)
        print(traceback.format_exc(), flush=True)
    yield
    # No cleanup needed — Pod gets torn down by RunPod when stopped


app = FastAPI(title="ERNIE-Image-Turbo Pod Server", lifespan=lifespan)


# ─── Pydantic schema ───
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    ratio: Optional[str] = None       # "16:9" | "1:1" | etc — overrides width/height
    width: Optional[int] = None
    height: Optional[int] = None
    steps: int = 8
    guidance_scale: float = 1.0
    seed: Optional[int] = None        # None or negative = random
    use_pe: bool = True               # Baidu's built-in prompt enhancer
    output_format: str = "jpg"        # "jpg" | "png"


# ─── Routes ───
@app.get("/health")
def health():
    return {
        "ready": state["pipe"] is not None,
        "model_id": MODEL_ID,
        "load_ms": state["load_ms"],
        "load_error": state["load_error"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _check_auth(authorization: Optional[str]):
    if not API_KEY:
        return  # No auth configured — URL is the secret
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <key>")
    if authorization.split(" ", 1)[1] != API_KEY:
        raise HTTPException(401, "Invalid token")


@app.post("/generate")
def generate(req: GenerateRequest, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    if state["pipe"] is None:
        raise HTTPException(503, f"Model not ready: {state['load_error'] or 'still loading'}")

    # Resolve size
    if req.ratio and req.ratio in RATIOS:
        w, h = RATIOS[req.ratio]
    elif req.width and req.height:
        w, h = int(req.width), int(req.height)
    else:
        w, h = RATIOS["16:9"]
    # Snap to /64 (diffusion model requirement)
    w = max(64, round(w / 64) * 64)
    h = max(64, round(h / 64) * 64)

    # Seed
    seed = req.seed if (req.seed is not None and req.seed >= 0) else random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # Generate
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
                use_pe=req.use_pe,
            )
        img: Image.Image = result.images[0]
        gen_ms = int((time.time() - t0) * 1000)

        # Encode
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
            "use_pe": req.use_pe,
            "generation_ms": gen_ms,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise HTTPException(507, f"CUDA OOM: {e}. Try a smaller resolution.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[server] generate ERROR:\n{tb}", flush=True)
        raise HTTPException(500, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    print(f"[server] Starting on {HOST}:{PORT}…", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
