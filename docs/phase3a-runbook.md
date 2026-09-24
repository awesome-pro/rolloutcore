# Phase 3A runbook: real vLLM control plane on one GPU

**Status: this runbook has been executed successfully.** The 2026-09-24 run is
recorded in `docs/phase3a-results.md` (16/16 checks, one A6000, the audited
commit). The notes below are what it took to get there.

**Goal.** Prove RolloutCore's lifecycle *control plane* against a real `vllm
serve`: fresh label, bootstrap to `rc-0`, a refused second bootstrap, deterministic
generation, `pause(mode="wait")` that genuinely waits for in-flight work and does
not block us, all three cache resets, paused-state validation, resume, and
deterministic generation again.

**Explicitly out of scope.** No NCCL, no weight replacement, no fabricated
no-op update. Phase 3A is an adapter-level integration test
(`scripts/live_control_plane_smoke.py`); the full nine-state cycle becomes real in
Phase 3C after Phase 3B implements the driver.

**Budget.** One small GPU, roughly 45 to 60 minutes of pod time. The harness itself
takes about a minute; the rest is environment setup.

---

## 0. Before you rent anything (free, on your machine)

```bash
cd /path/to/rolloutcore
./scripts/test.sh                 # tests + ruff + mypy + demo
VLLM_CHECKOUT=/path/to/vllm ./scripts/test.sh   # + anchor check
```

Then run the harness against the in-repo stub. This exercises the *entire*
sequence (all eleven checks, the JSON artifact, and the failure paths) with no
GPU and no vLLM:

```bash
PYTHONPATH=src:tests python3 tests/fake_dev_server.py --port 8123 &
python3 scripts/live_control_plane_smoke.py --base-url http://127.0.0.1:8123 \
    --model facebook/opt-125m --json-out /tmp/phase3a-stub.json
# expect: checks: N passed … RESULT: PASS, exit 0
kill %1
```

**Checkpoint CP0. Do not rent until:** `./scripts/test.sh` is green, the stub run
prints `RESULT: PASS`, and `pytest tests/test_live_smoke.py` is green (it covers
the drift, warm-server, and in-process-engine failure modes).

### Commit and push

The pod cannot see your machine. Either push the branch, or `rsync` it (step 2).

```bash
cd /path/to/rolloutcore
git add -A && git commit -m "Phase 3A: live control-plane harness, WeightProvenance rename"
git push origin main
```

---

## 1. Rent the pod

| Setting | Value | Why |
|---|---|---|
| GPUs | **1×** RTX 4090 / L40S / A6000 / A100 (≥16 GB), whichever is available | `opt-125m` needs ~1 GB; the harness is not compute-bound, so the GPU class is irrelevant to the result |
| Image | any recent **Ubuntu 24.04 + CUDA 12.8+** template, e.g. `runpod/pytorch:*-cu1281-torch280-ubuntu2404` | see the note below on why the template's torch version does not matter |
| Container disk | ≥ 50 GB (100 GB is fine) | image ~15 GB + model + pip/uv cache |
| Volume | **optional for a single 3A sitting** (RunPod's "nothing mounted" notice is a warning, not a blocker). Add one, mounted at `/workspace`, only if you may restart the pod or are continuing straight into 3B | keeps the vLLM install (~10 GB) and the HF cache across a restart; costs ~$0.07/GB/month whether or not the pod runs |
| Env var | `HF_HOME=/workspace/hf` **only if you mounted a volume** | so the model lands on the volume, not the ephemeral container |
| Exposed ports | `8888` (Jupyter) and `22` (SSH) only | **never expose 8000**: `/pause`, `/update_weight_version` and `/reset_*` are unauthenticated destructive controls |
| UDP | off | unused |

**The template's PyTorch version does not matter.** vLLM pins `torch==2.13.0`
(`requirements/cuda.txt:7`) and the wheel carries its own CUDA runtime, so
installing vLLM into a venv replaces whatever the image shipped. What matters is
the **host driver**, which `nvidia-smi` reports; that is what decides between
`cu129` and `cu130`.

**Don't bother with `vllm/vllm-openai` on RunPod** unless you are comfortable
overriding the container start command: its entrypoint launches the API server
itself, which fights the Jupyter/SSH setup a RunPod template gives you for free.
Installing vLLM into the RunPod image with `uv` (step 3, Option A) is *less*
work, not more.

### What the pinned commit actually requires

Read from a checkout at the audited commit, not guessed:

| Requirement | Value | Source |
|---|---|---|
| PyTorch | **2.13.0** (exact pin) | `requirements/cuda.txt:7` |
| CUDA wheel variants | **`cu129`** (default) and **`cu130`** | `setup.py:604` (`supported = {12: "cu129", 13: "cu130"}`) |
| Default CUDA for a source build | `VLLM_MAIN_CUDA_VERSION = "13.0"` | `vllm/envs.py:91` |
| Driver for `cu130` | **R580 or newer** (CUDA 13 minimum); `cu129` runs on far older drivers | `docs/getting_started/installation/gpu.cuda.inc.md:327` |
| Confirmed working (Phase 3A, 2026-09-24) | A6000 48 GB, driver **580.159.03**, CUDA **13.0**, Python 3.12.3, `pip install -U uv` → `cu130`, vLLM `0.30.1rc1.dev60+g00b7847c8`, torch `2.13.0+cu130` | `docs/phase3a-results.md` |
| Python | `>=3.10,<3.15` | `pyproject.toml:35` |
| Blackwell (B200/GB200) | needs ≥ CUDA 12.8 | `gpu.cuda.inc.md:39` |

So: **do not pick a torch version yourself**. vLLM pins it, and picking your own
is how these runs die. Pick the *variant* from the driver, which is what
`uv ... --torch-backend=auto` does for you.

An older "PyTorch 2.4 / CUDA 12.4" template (which this runbook previously
recommended) is ~9 torch minors and 9 CUDA minors behind this commit and will
not work.

Verify the driver before doing anything else:

```bash
nvidia-smi          # note the "CUDA Version:" field = the max CUDA this driver supports
```

- says **13.0+** → `cu130` wheels, or just let auto-detection decide
- says **12.x** → `cu129` wheels (`--torch-backend=cu129` / `VLLM_PRECOMPILED_WHEEL_VARIANT=cu129`)

**Checkpoint CP1. Proceed only if** `nvidia-smi` names the GPU. Record its
output; it goes into the report.

---

## 2. Get the code onto the pod

The repository is **private**, so a plain `https://` clone will stop at
`Username for 'https://github.com':`. Pick one:

**A. tar over SSH from your machine (no GitHub auth, and no `rsync` needed on the
pod; recommended).** In RunPod's Connect panel use the **SSH over exposed TCP**
tab, not the `ssh.runpod.io` proxy (that one documents "No support for SCP &
SFTP"):

```bash
# on your machine, in the checkout; substitute <host> and <port> from that tab
tar czf - -C "$(dirname "$(git rev-parse --show-toplevel)")" \
    --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache \
    --exclude=.mypy_cache --exclude=.ruff_cache --exclude=results \
    "$(basename "$(git rev-parse --show-toplevel)")" \
  | ssh -p <port> -i ~/.ssh/id_ed25519 root@<host> \
      'tar xzf - -C /workspace && ls /workspace/rolloutcore'
```

Keep `.git` in the copy: the artifact records `rolloutcore_sha` from
`git rev-parse HEAD` inside the repo. (~800 KB, so this is cheap.)

`rsync` is equivalent if the image has it, but note the SSH key must not be a
bare `~` inside `-e`, which rsync does not expand; use `$HOME`:

```bash
rsync -av -e "ssh -p <port> -i $HOME/.ssh/id_ed25519" \
    --exclude .venv --exclude __pycache__ --exclude results \
    /path/to/rolloutcore/ root@<host>:/workspace/rolloutcore/
```

**B. A personal access token** (fine for `git` commands, but do not paste the
token into any chat):

```bash
cd /workspace
git clone https://<TOKEN>@github.com/awesome-pro/rolloutcore.git && cd rolloutcore
```

A classic PAT needs `repo` scope; a fine-grained token needs *Contents: Read*.
The repository name is **`awesome-pro`**: a missing letter gives the credential
prompt too, so check it before assuming the token is wrong.

**C. Upload a tarball** through Jupyter's file browser, then
`tar xzf rolloutcore.tgz` in `/workspace`.

A tarball extracted as root keeps your machine's uid, so git refuses it with
`detected dubious ownership in repository`:

```bash
chown -R root:root /workspace/rolloutcore
# or, if you prefer not to chown:
#   git config --global --add safe.directory /workspace/rolloutcore
```

Then, whichever route you took:

```bash
cd /workspace/rolloutcore
python3 -V                        # >= 3.11
git rev-parse HEAD 2>/dev/null || echo "rsync copy: record your machine's HEAD instead"
ls src/rolloutcore scripts/live_control_plane_smoke.py tests/fake_dev_server.py
```

RolloutCore itself needs **nothing installed**: the harness is pure stdlib and
bootstraps `src/` onto `sys.path` by itself. Only vLLM needs installing.

---

## 3. Install vLLM pinned to the audited commit

Target: `00b7847c8036b667742b4efb21aab1de51fd4721`.

**Option 0 (you used the `vllm/vllm-openai` image):** nothing to install. Verify
and skip to step 4:

```bash
vllm --version && nvidia-smi
```

Otherwise, in a venv:

```bash
cd /workspace
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install uv        # uv is the supported installer here
```

**Option A (recommended): the prebuilt wheel for that exact commit.** vLLM
publishes a wheel per commit since v0.5.3:

```bash
export VLLM_COMMIT=00b7847c8036b667742b4efb21aab1de51fd4721
uv pip install vllm --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

!!! warning "`uv` must know the backend name"

    `--torch-backend` is an enum inside `uv`, and it lags the CUDA variants vLLM
    publishes. **uv 0.9.0 tops out at `cu129`** and rejects `cu130` outright
    (`invalid value 'cu130' for '--torch-backend'`), regardless of the driver.
    Either `uv self update` (or `pip install -U uv`) and retry `cu130`, or just
    install the **`cu129`** variant: a CUDA 12.9 build runs fine on a 580-series
    driver, which is newer than CUDA 12.9 requires. Confirmed working on
    uv 0.9.0 + driver 580.159 + A6000.

`--torch-backend=auto` reads the driver and picks the PyTorch index for you. If it
guesses wrong, name the variant explicitly:

```bash
# cu130 (needs a uv that supports it):
uv pip install vllm --torch-backend=cu130 \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cu130

# cu129 (works on uv 0.9.0, and on any driver that supports CUDA 12.9):
uv pip install vllm --torch-backend=cu129 \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cu129
```

`pip` is **not** supported against vLLM's nightly/commit indices (it merges
indexes and takes the newest version, so you silently get a different build). If
you insist on `pip`, install the wheel URL directly; see
`gpu.cuda.inc.md:67-72`.

**Option B: source at the pinned commit** (20 to 40 min, CUDA toolchain required;
use only if Option A fails):

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm && git fetch --depth=1 origin 00b7847c8036b667742b4efb21aab1de51fd4721
git checkout 00b7847c8036b667742b4efb21aab1de51fd4721
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

**Option C: latest release** (2 min). Acceptable, but the artifact's
`vllm_sha_matches_report` check then records a **warning**, and you must paste
that warning back with the results:

```bash
uv pip install vllm --torch-backend=auto
```

Then confirm the version **without importing vllm first**. An import failure
would otherwise hide which build you actually got:

```bash
python3 -c "import importlib.metadata as m; print('vllm', m.version('vllm')); print('torch', m.version('torch'))"
python3 -c "import torch; print('torch cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python3 -c "import vllm; print('vllm imports OK', vllm.__version__)"
python3 -c "import vllm.entrypoints.serve.dev.rlhf.api_router as r; print('dev router OK', len(r.router.routes))"
```

The metadata version must contain the commit (`g00b7847c8`). If it is a plain
release like `0.23.1`, then the commit index did not serve the request and uv
silently resolved against PyPI: you have the wrong build, and any CUDA-variant
mismatch will show up as an import error rather than a version mismatch.

**Variant consistency is mandatory.** The vLLM wheel and torch must come from the
same CUDA family. Mixing them produces
`ImportError: libcudart.so.13: cannot open shared object file` (a cu130 vLLM wheel
with cu129 torch) or the mirror image. Install both in one command from one
variant, and if you change your mind, recreate the venv rather than layering a
second install on top.

**Checkpoint CP2. Proceed only if** `vllm` imports and `vllm serve --help`
works. If you installed C instead of A/B, say so in the report; do not silently
mix versions.

---

## 4. Download the model

```bash
export HF_HOME=/workspace/hf
hf download facebook/opt-125m          # ~250 MB, matches vLLM's own RL example
# older CLIs: huggingface-cli download facebook/opt-125m
```

`opt-125m` is deliberate: it is what vLLM's `examples/rl/rlhf_http_nccl.py` uses,
so Phase 3B reuses the same model and removes model-specific unknowns. Qwen is a
later swap, not a Phase 3A concern.

**Checkpoint CP3.** `python3 -c "from transformers import AutoConfig;
print(AutoConfig.from_pretrained('facebook/opt-125m').model_type)"` → `opt`.

---

## 5. Start the server the way Phase 3A requires

Two env vars matter, and one of them is load-bearing:

```bash
cd /workspace/rolloutcore
export VLLM_SERVER_DEV_MODE=1            # exposes /pause, /weight_info, /reset_*
export VLLM_ENABLE_V1_MULTIPROCESSING=1  # engine core OUT of process
```

### tmux in one minute

The pod's terminal disconnects; a server started in a plain shell dies with it.

```
tmux new -s vllm        # create and attach a named session
Ctrl-b  d               # detach -> keeps running
tmux ls                 # list sessions
tmux attach -t vllm     # reattach
tmux kill-session -t vllm
```

Inside a session: `Ctrl-b c` new window, `Ctrl-b n` / `Ctrl-b p` next/previous,
`Ctrl-b 0` window 0, `Ctrl-b [` scroll mode (arrows, then `q` to exit).

**The prefix is a two-step chord, not a combination.** Press `Ctrl-b`, *release
both keys*, then press the next key on its own. Holding Ctrl through the second
key sends `Ctrl-c` instead, which kills whatever is in the foreground.

If the prefix keeps fighting your terminal, don't fight it:

- **A second SSH session is the simplest fix.** Keep the server in tmux window 0
  and open another terminal on your machine, `ssh` in again, and run the harness
  there. `tmux attach -t vllm` is only needed if you want to *watch* the server.
- **Or skip tmux for the server entirely**: `nohup` survives a disconnect just
  as well:

  ```bash
  nohup vllm serve facebook/opt-125m --host 127.0.0.1 --port 8000 \
      --enforce-eager --max-model-len 512 --gpu-memory-utilization 0.6 \
      > /workspace/server.log 2>&1 &
  tail -f /workspace/server.log        # watch startup, Ctrl-c to stop watching
  ```

- **Browsers eat `Ctrl-b`** (Firefox opens the bookmarks sidebar with it). If you
  are using RunPod's web terminal, switch to SSH from a real terminal, or rebind
  the prefix before starting tmux:

  ```bash
  printf 'set -g prefix C-a\nbind C-a send-prefix\n' > ~/.tmux.conf
  ```

Suggested layout: window 0 runs the server, window 1 runs the harness. Or use
`--launch` in step 6, which starts and stops the server itself; still do it
inside tmux so an accidental disconnect cannot kill the run mid-drain.

`VLLM_ENABLE_V1_MULTIPROCESSING=1` is the default, but say it explicitly: the
in-process engine path rejects `mode="wait"` outright
(`ValueError`, `vllm/v1/engine/core.py:902`), and `mode="wait"` is the only drain
mode RolloutCore accepts: `mode="keep"` lets one response span two weight
versions.

Manual start (useful for the first run; `--launch` in step 6 does this for you).
Run it under `tmux`: a web terminal disconnect should not kill the server, and
`--launch` mode needs no second shell at all:

```bash
tmux new -s vllm
export HF_HOME=/workspace/hf
vllm serve facebook/opt-125m \
    --host 127.0.0.1 --port 8000 \
    --enforce-eager \
    --max-model-len 512 \
    --gpu-memory-utilization 0.6
# Ctrl-b d to detach; `tmux attach -t vllm` to return
```

Quick manual pre-check in a second shell:

```bash
curl -s localhost:8000/weight_info   # {"weight_version":"default"}
curl -s localhost:8000/is_paused     # {"is_paused":false}
curl -s localhost:8000/get_world_size
```

**Checkpoint CP4. Proceed only if** `/weight_info` says `"default"`. If it says
anything `rc-*`, the server is not fresh: restart it (or use
`--reset-label-to-default`, which is a dev-mode escape hatch, not a workflow).

---

## 6. Run the smoke

Attach mode (server already running):

```bash
cd /workspace/rolloutcore
python3 scripts/live_control_plane_smoke.py \
    --base-url http://127.0.0.1:8000 \
    --model facebook/opt-125m \
    --vllm-sha 00b7847c8036b667742b4efb21aab1de51fd4721 \
    --identity-source manifest \
    --json-out results/phase3a.json
```

Launch mode (one command, no second shell):

```bash
cd /workspace/rolloutcore
python3 scripts/live_control_plane_smoke.py \
    --launch --model facebook/opt-125m \
    --vllm-sha 00b7847c8036b667742b4efb21aab1de51fd4721 \
    --identity-source manifest
```

`--identity-source manifest` loads the model client-side (125M params, seconds)
and hashes its real parameter manifest into the bootstrap `WeightIdentity`. That
is the honest identity for I4; `--identity-source declared` (the default) records
only `model@revision` and is fine when torch/transformers are unavailable.

The run takes ~1 minute and prints one line per check:

```
  [PASS] environment
  [PASS] server_healthy
  [PASS] fresh_weight_info_is_default
  [PASS] control_plane_bootstrap_seeds_rc0
  [PASS] second_bootstrap_refused_without_write
  [PASS] deterministic_generation_before_pause
  [PASS] rollout_binding_carries_identity
  [PASS] inflight_request_started
  [PASS] pause_is_nonblocking_on_the_rolloutcore_side
  [PASS] pause_waits_for_inflight_request
  [PASS] engine_reports_paused
  [PASS] cache_resets_succeed
  [PASS] pre_resume_validation_sees_rc0_and_paused
  [PASS] resume_succeeds
  [PASS] deterministic_generation_after_resume
  [PASS] controller_tainted_by_design
   checks: 16 passed, 0 warned, 0 failed
   RESULT: PASS
```

**The controller ends `TAINTED` on purpose.** Phase 3A resumes the engine through
the adapter, outside the controller, because V1's table has no update-free
revalidation path (`docs/state-machine.md` §12 item 5). A controller that cannot
vouch for a serving engine must not pretend otherwise, so it taints and the
report marks it `expected`. That is the fail-closed behaviour, not a failure.

**Checkpoint CP5. Phase 3A is done when** the script exits `0` with
`"failed_checks": []`. A `warn` on `environment` means the vLLM version is not the
audited commit; record it and continue, but say so.

---

## 7. Package the artifacts and stop the pod

```bash
cd /workspace/rolloutcore
cat results/phase3a.json                     # paste this back verbatim
nvidia-smi > results/gpu.txt
python3 -c "import vllm; print(vllm.__version__, vllm.__file__)" | tee results/vllm-version.txt
pip freeze | grep -Ei "vllm|torch|transformers|flashinfer" | tee results/pip.txt
tar czf /workspace/phase3a-results.tgz results/
```

`results/` is gitignored; if you want the artifact in the repository, force it:

```bash
git add -f results/phase3a.json
```

Download `phase3a-results.tgz` (RunPod file browser, `scp`, or `runpodctl`), then
**stop the pod.** Phase 3A needs one GPU for under an hour; there is no reason to
keep paying while the results are reviewed.

Without a volume: **nothing on the pod survives `Stop`** (the container disk is
erased), so pull `results/phase3a.json` *before* stopping; step 7 does exactly
that. Re-running later costs a fresh `uv pip install` (~5 min) and a 250 MB model
download.

**Checkpoint CP6. Stop here.** Send back:

1. `results/phase3a.json` (all of it),
2. the `gpu.txt` / `vllm-version.txt` / `pip.txt` lines,
3. the server log tail if anything failed,
4. any `WARN` line verbatim.

Do **not** start Phase 3B in the same session. Phase 3B needs two GPUs and a
different failure surface (NCCL/rendezvous), and its plan depends on what Phase 3A
actually showed.

---

## Pass/fail criteria

| Check | Expected on a healthy run |
|---|---|
| `fresh_weight_info_is_default` | `"default"` exactly |
| `control_plane_bootstrap_seeds_rc0` | `pre_seed="default"` → `"rc-0"`, `driver=None`, no transfer engine |
| `second_bootstrap_refused_without_write` | `AlreadyManagedEngineError`, label still `rc-0`, only `GET /weight_info` sent |
| `deterministic_generation_before_pause` | two greedy requests, identical text **and** token ids |
| `pause_is_nonblocking_on_the_rolloutcore_side` | `begin_drain` and first `await_drain` each < 0.5 s |
| `pause_waits_for_inflight_request` | full token count, then drain completes; `engine_drain_completed=False` on the first poll |
| `engine_reports_paused` | `is_paused=true`, label still `rc-0` |
| `cache_resets_succeed` | `prefix=true encoder=true mm=true`, raw statuses all `< 400` |
| `pre_resume_validation_sees_rc0_and_paused` | `weight_version="rc-0"`, `is_paused=true` |
| `resume_succeeds` | acknowledged, `is_paused=false` (confirmed twice) |
| `deterministic_generation_after_resume` | identical to the pre-pause output |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pause_mode_wait_supported` fails with `'wait' mode can't be used in inproc-engine mode` | engine core is in-process | `VLLM_ENABLE_V1_MULTIPROCESSING=1` (and do not pass `--disable-frontend-multiprocessing`) |
| `/weight_info` → 404 | dev mode off | `VLLM_SERVER_DEV_MODE=1` |
| `fresh_weight_info_is_default` fails with `rc-…` | server already driven by a previous run | restart the server, or `--reset-label-to-default` (dev only) |
| `deterministic_generation_after_resume` fails | genuine incoherence, or kernel nondeterminism | first re-run; if it reproduces, keep the failure and report it. That is exactly the kind of finding Phase 3A exists to surface. `--no-strict-generation` records it as a warning instead of failing |
| `cache_resets_succeed` fails on prefix | blocks still held | the drain must complete first; check the pause actually returned |
| CUDA OOM at startup | `--max-model-len`/`--gpu-memory-utilization` too high for the pod | lower both; 512 / 0.6 is already conservative |
| `connection refused` on `/health` | server still loading or crashed | the launcher waits up to 900 s and prints the log tail on failure |
| `environment` warns | vLLM is not the audited commit | record it in the report; do not hide it |
| `ImportError: libcudart.so.13` (or `.so.12`) | vLLM wheel and torch came from different CUDA variants | `rm -rf .venv`, recreate it, and install both from one variant in a single `uv pip install`; never layer a second install over a mismatched one |
| `invalid value 'cu130' for '--torch-backend'` | `--torch-backend` is an uv enum, and uv 0.9.0 stops at `cu129` | `pip install -U uv`, or use `cu129` (fine on a 580 driver) |
| `fatal: detected dubious ownership` | a tarball copied from your machine carried uid 501 into a root shell | `chown -R root:root /workspace/rolloutcore`, or add a `safe.directory` exception |
| `CUDA driver version is insufficient for CUDA runtime version` | the wheel is `cu130` but the driver is older than R580 (`gpu.cuda.inc.md:327`) | reinstall with `--torch-backend=cu129`, or move to a pod whose driver is R580+ |
| torch ended up at the wrong version | it was installed by hand, or by `pip` against a nightly index | never pin torch yourself: it is pinned at `2.13.0` by `requirements/cuda.txt:7`. Reinstall with `uv` (step 3) |
| an old image ships PyTorch 2.4 / CUDA 12.4 | stale template, ~9 minors behind this commit | use `vllm/vllm-openai`, or install vLLM per step 3; the template's torch does not matter once `uv --torch-backend=auto` runs |
| `No module named 'vllm.entrypoints.serve.dev'` | a vLLM older than the dev-route layout | pin to Option A/B; this layout is what the audit was done against |

## What Phase 3A does *not* prove

- That weights can be replaced. No NCCL driver exists yet, so that is Phase 3B.
- That `WeightIdentity` matches what the engine is holding: the engine reports
  only an opaque version string, so the identity is *declared* provenance, not
  verified bytes.
- That a rollout cannot span two versions: that needs a real update (Phase 3B/3C)
  and is where the `mode="wait"` drain choice earns its keep.
