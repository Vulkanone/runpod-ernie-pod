/**
 * ERNIE-Image-Turbo Pod Client (HTTP).
 *
 * Calls a long-running FastAPI server on a RunPod Pod (not Serverless).
 * Public URL format: https://<pod_id>-<port>.proxy.runpod.net
 *
 * If POD_API_KEY is set in .env, sends `Authorization: Bearer <key>`.
 */

const DEFAULT_RATIOS = {
  "1:1":  { width: 1024, height: 1024 },
  "16:9": { width: 1376, height: 768 },
  "9:16": { width: 768, height: 1376 },
  "4:3":  { width: 1200, height: 896 },
  "3:4":  { width: 896, height: 1200 },
  "3:2":  { width: 1264, height: 848 },
  "2:3":  { width: 848, height: 1264 },
};

export class ErniePodClient {
  constructor({ baseUrl, apiKey = null, verbose = false } = {}) {
    if (!baseUrl) throw new Error("baseUrl is required (POD_URL)");
    this.baseUrl = baseUrl.replace(/\/+$/, ""); // strip trailing slashes
    this.apiKey = apiKey;
    this.verbose = verbose;
  }

  log(...args) {
    if (this.verbose) console.log("[pod]", ...args);
  }

  _headers() {
    const h = {
      "Content-Type": "application/json",
      "Accept": "application/json",
      "User-Agent": "runpod-ernie-pod-client/1.0",
    };
    if (this.apiKey) h["Authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  /** Quick health-check — returns the server's /health JSON. */
  async health(timeoutMs = 10_000) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        headers: this._headers(),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } finally {
      clearTimeout(t);
    }
  }

  /**
   * Generate 1 image. options keys mirror the Pydantic schema on the server.
   * Returns: { jpegBuffer, mimeType, width, height, seed, generationMs, …, prompt }
   */
  async generateImage(options) {
    if (!options?.prompt) throw new Error("prompt is required");

    const body = {
      prompt: options.prompt,
      negative_prompt: options.negativePrompt || null,
      num_inference_steps: options.steps ?? 8,
      guidance_scale: options.guidanceScale ?? 1.0,
      seed: options.seed ?? -1,
      use_pe: options.usePe !== false,
      output_format: options.outputFormat || "jpg",
    };
    if (options.ratio && DEFAULT_RATIOS[options.ratio]) {
      const r = DEFAULT_RATIOS[options.ratio];
      body.width = r.width;
      body.height = r.height;
      body.ratio = options.ratio;
    } else if (options.width && options.height) {
      body.width = options.width;
      body.height = options.height;
    }

    const timeoutMs = options.timeoutMs || 120_000;
    const ctrl = new AbortController();
    const tHandle = setTimeout(() => ctrl.abort(), timeoutMs);

    this.log(`POST /generate prompt="${options.prompt.slice(0, 60)}…" size=${body.width || "?"}x${body.height || "?"}`);
    const t0 = Date.now();
    let res;
    try {
      res = await fetch(`${this.baseUrl}/generate`, {
        method: "POST",
        headers: this._headers(),
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
    } catch (err) {
      clearTimeout(tHandle);
      if (err.name === "AbortError") throw new Error(`Timeout after ${timeoutMs}ms`);
      throw err;
    }
    clearTimeout(tHandle);

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Pod ${res.status}: ${text.slice(0, 300)}`);
    }
    const data = await res.json();
    if (!data.image_base64) {
      throw new Error(`No image_base64 in response: ${JSON.stringify(data).slice(0, 300)}`);
    }

    return {
      jpegBuffer: Buffer.from(data.image_base64, "base64"),
      mimeType: data.format === "png" ? "image/png" : "image/jpeg",
      format: data.format,
      width: data.width,
      height: data.height,
      seed: data.seed,
      steps: data.steps,
      guidanceScale: data.guidance_scale,
      usePe: data.use_pe,
      generationMs: data.generation_ms,
      totalMs: Date.now() - t0,
      prompt: options.prompt,
    };
  }
}
