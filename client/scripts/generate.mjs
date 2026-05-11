#!/usr/bin/env node
/**
 * Single-image generator against the ERNIE Pod server.
 *
 * Usage:
 *   node scripts/generate.mjs -p "<prompt>" -r 16:9
 *
 * Reads POD_URL + POD_API_KEY from .env (or env vars).
 */

import { writeFileSync, mkdirSync, existsSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { parseArgs } from "util";
import { ErniePodClient } from "./lib/pod-client.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(__dirname, "..");

function loadDotenv() {
  const envPath = join(PROJECT_ROOT, ".env");
  if (!existsSync(envPath)) return {};
  const env = {};
  for (const line of readFileSync(envPath, "utf-8").split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const eq = t.indexOf("=");
    if (eq < 0) continue;
    const k = t.slice(0, eq).trim();
    let v = t.slice(eq + 1).trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    env[k] = v;
  }
  return env;
}
const env = { ...loadDotenv(), ...process.env };

const { values } = parseArgs({
  options: {
    prompt:   { type: "string",  short: "p" },
    ratio:    { type: "string",  short: "r", default: "16:9" },
    output:   { type: "string",  short: "o", default: "./output" },
    steps:    { type: "string",  default: "8" },
    guidance: { type: "string",  default: "1.0" },
    seed:     { type: "string" },
    format:   { type: "string",  default: "jpg" },
    "no-pe":  { type: "boolean", default: false },
    timeout:  { type: "string",  default: "120" },
    verbose:  { type: "boolean", short: "v", default: false },
    help:     { type: "boolean", short: "h", default: false },
  },
});

if (values.help || !values.prompt) {
  console.log(`ERNIE-Image-Turbo Pod — generate one image

Usage: node scripts/generate.mjs -p "<prompt>" [opts]

Options:
  -p, --prompt    Required
  -r, --ratio     "1:1" | "4:3" | "3:4" | "16:9" | "9:16" | "3:2" | "2:3"   [default 16:9]
  -o, --output    Output dir                                                  [default ./output]
      --steps     Inference steps (8 is Baidu's recommended)                  [default 8]
      --guidance  Guidance scale (1.0 = no CFG, Baidu's default)              [default 1.0]
      --seed      Integer seed (omit for random)
      --format    "jpg" | "png"                                               [default jpg]
      --no-pe     Disable Baidu's built-in prompt enhancer (faster)
      --timeout   Per-call timeout in seconds                                 [default 120]
  -v, --verbose

Env (.env):
  POD_URL         e.g. https://abc12def-8000.proxy.runpod.net
  POD_API_KEY     optional Bearer token
`);
  process.exit(values.help ? 0 : 1);
}

if (!env.POD_URL) {
  console.error("ERROR: POD_URL not set. Edit .env (RunPod Pod proxy URL).");
  process.exit(1);
}
if (!existsSync(values.output)) mkdirSync(values.output, { recursive: true });

const client = new ErniePodClient({
  baseUrl: env.POD_URL,
  apiKey: env.POD_API_KEY || null,
  verbose: values.verbose,
});

console.log(`Pod: ${env.POD_URL}`);
console.log(`Prompt: "${values.prompt.slice(0, 90)}${values.prompt.length > 90 ? "…" : ""}"`);
console.log(`Ratio: ${values.ratio}, steps: ${values.steps}, guidance: ${values.guidance}\n`);

try {
  // Quick health probe first
  if (values.verbose) {
    const h = await client.health();
    console.log(`Health: ready=${h.ready} model=${h.model_id} gpu=${h.gpu}`);
    if (!h.ready) {
      console.error(`Server not ready: ${h.load_error || "still loading"}`);
      process.exit(2);
    }
  }

  const r = await client.generateImage({
    prompt:        values.prompt,
    ratio:         values.ratio,
    steps:         parseInt(values.steps),
    guidanceScale: parseFloat(values.guidance),
    seed:          values.seed ? parseInt(values.seed) : undefined,
    usePe:         !values["no-pe"],
    outputFormat:  values.format,
    timeoutMs:     parseInt(values.timeout) * 1000,
  });

  const ts = Date.now();
  const ext = r.mimeType === "image/png" ? "png" : "jpg";
  const filepath = join(values.output, `ernie_${ts}_${r.seed}.${ext}`);
  writeFileSync(filepath, r.jpegBuffer);
  console.log(`OK — total ${(r.totalMs / 1000).toFixed(1)}s (gen ${r.generationMs}ms)  ${r.width}×${r.height}  seed=${r.seed}`);
  console.log(`     → ${filepath}`);
} catch (e) {
  console.error(`FAIL: ${e.message}`);
  process.exit(1);
}
