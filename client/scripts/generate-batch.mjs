#!/usr/bin/env node
/**
 * Batch generator against the ERNIE Pod server.
 *
 * Input JSON shape (grok-it / freegen-it compatible):
 *   { "prompts": [ { "text": "...", "name": "img_001" }, ... ] }
 *
 * The Pod serves requests sequentially (1 GPU = 1 generation at a time).
 * Concurrency on the CLIENT side > 1 means we queue requests on the server
 * — but only 1 generates at a time. So concurrency > 1 is only useful if
 * you scale the Pod horizontally (multiple Pods behind a load balancer);
 * for a single Pod, stick with concurrency 1.
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
    file:        { type: "string",  short: "f" },
    ratio:       { type: "string",  short: "r", default: "16:9" },
    output:      { type: "string",  short: "o", default: "./output" },
    concurrency: { type: "string",  short: "c", default: "1" },
    steps:       { type: "string",  default: "8" },
    guidance:    { type: "string",  default: "1.0" },
    format:      { type: "string",  default: "jpg" },
    "no-pe":     { type: "boolean", default: false },
    timeout:     { type: "string",  default: "180" },
    verbose:     { type: "boolean", short: "v", default: false },
    help:        { type: "boolean", short: "h", default: false },
  },
});

if (values.help || !values.file) {
  console.log(`ERNIE Pod — Batch generator (grok-it compatible I/O)

Usage: node scripts/generate-batch.mjs -f prompts.json [opts]

Options:
  -f, --file         JSON: { prompts: [{text, name}, ...] }       [required]
  -r, --ratio        16:9, 1:1, 9:16, 4:3, 3:4, 3:2, 2:3          [default 16:9]
  -o, --output       Output dir                                    [default ./output]
  -c, --concurrency  Parallel requests (only useful with multiple
                     Pods; single Pod = 1 GPU = no parallel gain) [default 1]
      --steps        Inference steps                               [default 8]
      --guidance     Guidance scale                                [default 1.0]
      --format       jpg | png                                     [default jpg]
      --no-pe        Disable Baidu's prompt enhancer
      --timeout      Per-call timeout (sec)                        [default 180]
  -v, --verbose
`);
  process.exit(values.help ? 0 : 1);
}

if (!env.POD_URL) {
  console.error("ERROR: POD_URL not set in .env");
  process.exit(1);
}
if (!existsSync(values.file)) {
  console.error(`File not found: ${values.file}`);
  process.exit(1);
}
const input = JSON.parse(readFileSync(values.file, "utf-8"));
if (!Array.isArray(input.prompts) || input.prompts.length === 0) {
  console.error('Input must be {"prompts": [{text, name}, ...]}');
  process.exit(1);
}
if (!existsSync(values.output)) mkdirSync(values.output, { recursive: true });

const client = new ErniePodClient({
  baseUrl: env.POD_URL,
  apiKey: env.POD_API_KEY || null,
  verbose: values.verbose,
});

const concurrency = Math.max(1, Math.min(10, parseInt(values.concurrency) || 1));
const steps = parseInt(values.steps);
const guidance = parseFloat(values.guidance);
const timeoutMs = parseInt(values.timeout) * 1000;

console.log(`Pod: ${env.POD_URL}`);
console.log(`Prompts: ${input.prompts.length} · ratio ${values.ratio} · steps ${steps} · guidance ${guidance}`);
console.log(`Concurrency: ${concurrency}\n`);

const results = [];
const start = Date.now();

async function processOne(p) {
  const { text, name } = p;
  if (!text || !name) return { name: name || "(no-name)", success: false, error: "Missing text/name" };
  const t0 = Date.now();
  try {
    const r = await client.generateImage({
      prompt:        text,
      ratio:         values.ratio,
      steps,
      guidanceScale: guidance,
      usePe:         !values["no-pe"],
      outputFormat:  values.format,
      timeoutMs,
    });
    const ext = r.mimeType === "image/png" ? "png" : "jpg";
    const filename = `${name}.${ext}`;
    writeFileSync(join(values.output, filename), r.jpegBuffer);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`[pod] OK   ${elapsed}s (gen ${r.generationMs}ms)  ${name}  "${text.slice(0, 60)}${text.length > 60 ? "…" : ""}"`);
    return { name, success: true, file: filename };
  } catch (err) {
    const msg = err?.message || String(err);
    console.error(`[pod] FAIL  ${name}  ${msg.slice(0, 120)}`);
    return { name, success: false, error: msg };
  }
}

async function runPool() {
  const queue = input.prompts.slice();
  async function worker() {
    while (queue.length > 0) {
      const p = queue.shift();
      const r = await processOne(p);
      results.push(r);
    }
  }
  await Promise.all(Array.from({ length: concurrency }, () => worker()));
}

await runPool();

const order = new Map(input.prompts.map((p, i) => [p.name, i]));
results.sort((a, b) => (order.get(a.name) ?? 0) - (order.get(b.name) ?? 0));

const resultsPath = join(values.output, "results.json");
writeFileSync(resultsPath, JSON.stringify({ results }, null, 2));

const okCount = results.filter((r) => r.success).length;
const failCount = results.length - okCount;
const totalElapsed = ((Date.now() - start) / 1000).toFixed(1);
console.log(`\n[pod] Done in ${totalElapsed}s — ${okCount}/${results.length} ok, ${failCount} failed`);
console.log(`[pod] Results: ${resultsPath}`);
process.exit(0);
