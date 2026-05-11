# runpod-ernie-pod

ERNIE-Image-Turbo on a **RunPod Pod** (NOT Serverless) + Node client.

A Pod is a long-running GPU VM you rent by the hour. We launch a FastAPI
server (`server.py`) inside it that loads the model once and serves
generations over HTTP. No cold starts after the initial load, no
Serverless build pipeline drama.

## Repo layout

```
runpod-ernie-pod/
├── server.py           ← FastAPI HTTP server. Runs INSIDE the Pod.
├── requirements.txt    ← Python deps. Installed by start.sh.
├── start.sh            ← One-shot installer + launcher.
├── client/             ← Node CLI client. Runs on YOUR laptop.
│   ├── scripts/generate.mjs
│   ├── scripts/generate-batch.mjs
│   ├── scripts/lib/pod-client.mjs
│   ├── .env.example
│   └── package.json
└── README.md
```

## End-to-end setup

### 0. Push this repo to GitHub
```bash
cd D:/Programs/runpod-ernie-pod
git init && git add . && git commit -m "Initial Pod server + Node client"
git branch -M main
# Create empty repo on github.com first, then:
git remote add origin https://github.com/<your-user>/runpod-ernie-pod.git
git push -u origin main
```

### 1. Launch a RunPod Pod
1. https://www.runpod.io/console/pods → **+ Deploy**
2. **GPU**: **A40 48GB** community (~$0.40/h) — sweet spot for ERNIE-8B
3. **Template**: pick one with PyTorch + CUDA pre-installed, e.g.:
   - `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
4. **Container Disk**: 30 GB (room for HF cache)
5. **Volume Mount**: 20+ GB at `/workspace` (persistent across stop/start)
6. **Expose HTTP Port**: 8000 (this is what the FastAPI server listens on)
7. Deploy. Wait ~30-60s for the Pod to boot.

### 2. SSH or use Web Terminal
Once the Pod shows status **Running**, click **Connect** → either:
- **Web Terminal** (in-browser, easiest) — opens a shell tab
- **SSH** (if you've uploaded your SSH key in RunPod settings)

### 3. Clone + run
Inside the Pod's terminal:
```bash
cd /workspace
git clone https://github.com/<your-user>/runpod-ernie-pod.git
cd runpod-ernie-pod
bash start.sh
```

What happens:
- Installs `git`, pip deps (~2-3 min)
- Downloads ERNIE-Image-Turbo from HF (~16GB, ~3-5 min first time)
- Launches FastAPI on port 8000
- Prints `[server] ✓ Model ready in X.Xs — ready to serve`

Leave this terminal open (the server runs in foreground). If you close
the terminal the server dies. To run in background, see "Persistence" below.

### 4. Get the public URL
On the RunPod Pod page → **Connect** → **HTTP Service [Port 8000]**.
RunPod gives you a URL like:
```
https://abc12def34-8000.proxy.runpod.net
```
Copy this URL.

### 5. Configure the local Node client
On YOUR laptop:
```bash
cd D:/Programs/runpod-ernie-pod/client
cp .env.example .env
notepad .env
```
Paste:
```
POD_URL=https://abc12def34-8000.proxy.runpod.net
POD_API_KEY=
```

### 6. Test
```bash
node scripts/generate.mjs -p "vintage 1950s painted comic illustration of a diner at sunset" -r 16:9 -v
```
Expected output (~5-10s warm):
```
Pod: https://abc12def34-8000.proxy.runpod.net
Prompt: "vintage 1950s painted…"
Health: ready=true model=Baidu/ERNIE-Image-Turbo gpu=NVIDIA A40
OK — total 4.3s (gen 3850ms)  1376×768  seed=1234567
     → output/ernie_<ts>_<seed>.jpg
```

### 7. Batch
```bash
echo '{"prompts":[
  {"text":"red apple on table","name":"img_001"},
  {"text":"green forest at dawn","name":"img_002"},
  {"text":"vintage diner at sunset","name":"img_003"}
]}' > prompts.json
node scripts/generate-batch.mjs -f prompts.json -o ./output/batch1 -r 16:9 --concurrency 1
```
Single Pod = 1 GPU = sequential generation. ~4-5s per image.

## Costs

A40 community @ ~$0.40/h:
- 10-min session generating 100 images → **$0.07**
- 1-hour session generating 600 images → **$0.40**
- Pod left running 24/7 → **$288/month** (don't do this unless generating 24/7)

**Always stop the Pod when done** to avoid idle charges.

## Stop/restart the Pod

- **Stop**: console → pod → ⋮ → Stop. Volume `/workspace` keeps your files
  + HF model cache. Compute stops billing.
- **Restart**: ⋮ → Start. Pod boots in ~30s, HF cache restored, model
  loads from local disk in ~15-30s (no re-download).

## Persistence (server keeps running if you close terminal)

Replace the last command in start.sh with:
```bash
nohup python server.py > /workspace/server.log 2>&1 &
```
Then `tail -f /workspace/server.log` to monitor. To stop: `pkill -f server.py`.

Or use `tmux`/`screen`:
```bash
tmux new -s ernie
bash start.sh
# Detach: Ctrl+B then D. Reattach: tmux attach -t ernie
```

## Optional: auth on the server

If you're worried someone might find your URL and spam it (unlikely — URLs
are random 11-char ids), set a token before launching:
```bash
export POD_API_KEY="some-random-token-here"
bash start.sh
```
And put the same token in `client/.env` → `POD_API_KEY=some-random-token-here`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `pip install` fails on diffusers | git missing | start.sh installs it via apt — re-run |
| `OutOfMemoryError` in /generate | A40 stretched — usually fine | Try ratio 1:1 instead of 16:9 |
| `503 Model not ready` | Still loading | Wait, retry in 30s |
| `Connection refused` | Server not started or wrong port | Check `bash start.sh` is running |
| `401 Missing Bearer` | POD_API_KEY mismatch | Match .env to server env |
| Pod stuck "starting" >5min | RunPod backend slow | Stop + start the pod |
