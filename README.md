# ollama-rocm-gfx1103-ubuntu

Native ROCm acceleration for [Ollama](https://github.com/ollama/ollama) on the AMD **Radeon 780M iGPU** (`gfx1103`, the integrated GPU in Ryzen 7040 / 8040 / Phoenix series), on Ubuntu Linux.

This repo is a one-command setup that automates a process documented in detail in [this blog post](https://ataary.com/ubuntu-linux-with-ollama-rocm-on-amd-ryzen-780m-igpu/). If you've got a Phoenix iGPU and want to run local LLMs on it without falling back to CPU, this is for you.

## What it does

Native ROCm `gfx1103` support on Linux is missing from every standard place — Ubuntu's `librocblas5`, AMD's official ROCm .debs, and Ollama's bundled libraries all ship kernels for surrounding architectures (gfx1030/1100/1101/1102/1151/1200) but skip Phoenix. The result is that Ollama enumerates the iGPU as a ROCm device, then crashes when it tries to actually run a GEMM operation, and silently falls back to CPU.

The setup script in this repo closes the gap by:

1. Building [`likelovewant/ollama-for-amd`](https://github.com/likelovewant/ollama-for-amd) from source with `-DAMDGPU_TARGETS=gfx1103`.
2. Applying three small patches to `ml/device.go` — see [`patches/`](patches/).
3. Downloading Fedora 44's `rocblas-7.1.1-7.fc44.x86_64.rpm` and extracting just the `gfx1103` Tensile kernel files (the actual GPU bytecode) into the system rocBLAS library directory. The kernel binaries are distribution-agnostic at the binary level — Fedora's compile, Ubuntu's runtime, no problem **as long as the kernel ABI matches the runtime ABI**. Earlier versions of this repo used Fedora 43's rocBLAS 6.4 kernels; those run but crash often under 7.1 runtime (see the **Stability** section below).
4. Writing a systemd drop-in so the resulting setup survives reboot, disables Flash Attention (required — see Stability), enables a long keep-alive, and (optionally) pins Ollama to a specific GPU on multi-GPU systems.

After running, Ollama reports `library=ROCm compute=gfx1103` and inference runs at ~12–48 tokens/sec on `gemma4:e2b`/`e4b` depending on prompt size (vs ~16 tok/s CPU-only).

## Stability — what to actually expect

**With Ubuntu HWE kernel `7.0.0-15` or newer, the conservative env-var config below, and (if you're using Open WebUI) the correct OWUI bypass setting, this setup is reliable enough for daily use.** Latest measurement: **52/52 successful OWUI web-search chat completions** in a real-world session, zero MES events.

The history of this number:

- **~42% crash rate** = Fedora 6.4 rocBLAS kernels under Ubuntu's rocBLAS 7.1 runtime. Fix: re-run `setup.sh` (it now uses matching Fedora 7.1.1 kernels).
- **~14% crash rate** = correct kernels but kernel ≤ `7.0.0-14`. Fix: `sudo apt install linux-image-generic-hwe-26.04 && sudo reboot`.
- **~30% crash rate** specifically with OWUI web search, even on kernel `-15` with the env-var config below = Ollama Cloud Search returning full Wikipedia articles (~41k tokens per result) and OWUI passing them through unchunked, producing 150k+ token prompts that get truncated to 16k *and* exercise the residual MES issue at scale. Fix: see [Using with Open WebUI](#using-with-open-webui) below — flip OWUI's `bypass_embedding_and_retrieval` to **False** so it chunks + retrieves top-k instead.
- **0% crash rate observed** = all of the above in place. The set of fixes is non-obvious because the failure modes look identical (the same "ROCm error: unspecified launch failure" can come from any of them). All four layers need to be right.

### What "conservative env-var config" means

The default Ollama settings (`OLLAMA_NUM_BATCH=512` and 4k default context) are fine *if your workload is tiny*, but pushing context up or running a higher batch size makes the compute graph large enough that the residual MES susceptibility comes back. The combination that works reliably:

```ini
Environment="OLLAMA_FLASH_ATTENTION=0"      # disable FA; still brittle on gfx1103
Environment="OLLAMA_CONTEXT_LENGTH=16384"   # 16k, NOT 32k or higher
Environment="OLLAMA_NUM_BATCH=256"          # 256, NOT the default 512
Environment="OLLAMA_KEEP_ALIVE=24h"         # keep models resident
Environment="OLLAMA_MAX_LOADED_MODELS=1"    # one model at a time
```

The `override.conf` template in this repo ships these defaults.

### The dmesg signature to watch for

If crashes do happen on `-15` with these settings, check `sudo dmesg | grep MES`. The bug looks like:

```
amdgpu: MES failed to respond to msg=REMOVE_QUEUE
amdgpu: MES might be in unrecoverable state, issue a GPU reset
amdgpu: GPU reset succeeded, trying to resume
```

If you see these messages clustered around crash times, you're hitting the residual MES issue. Most common cause if your env-vars are correct: OWUI sending a giant prompt (see next section). Less common: pushing context above 16k or batch above 256 for some workload. The mitigation in either case is to make the compute graph smaller.

### Using with Open WebUI

If you're consuming this Ollama instance from [Open WebUI](https://github.com/open-webui/open-webui) with web search enabled, **there are two OWUI-side settings that matter as much as the Ollama-side env-vars above** — and one of them is non-obvious:

| OWUI Setting (Admin Panel → Settings → Web Search) | Value | Why |
|---|---|---|
| `bypass_web_loader` | **True** | Skip the langchain HTML scrape step. Without this, OWUI re-fetches every URL through a default user-agent that Cloudflare-protected sites (egpu.io, dev.to, etc.) block. Symptom: model says "I see N sources but no content." |
| `bypass_embedding_and_retrieval` | **False** ← counterintuitive! | Earlier guidance said True. **Wrong for most search engines.** Ollama Cloud Search returns full page content (Wikipedia articles = ~41k tokens per result). With `bypass=True`, OWUI shoves all of it (3 query variants × 3 results = ~150k tokens) directly into the prompt; Ollama truncates to 16k, dropping most of the actual relevant content, and the heavy compute graph triggers MES failures. With `bypass=False`, OWUI chunks the content (chunk_size=1000), embeds on CPU using `sentence-transformers/all-MiniLM-L6-v2`, and ships only the top-k most relevant chunks (~3k tokens) to the model. Same answer quality, ~50× smaller prompt, dramatically more stable. |

To change `bypass_embedding_and_retrieval` from True → False reliably (the UI toggle can be overwritten by OWUI's in-memory state on restart), stop the container first:

```bash
sudo docker stop open-webui

# patch the setting in the SQLite blob while the container isn't running
sudo docker run --rm -v open-webui:/data alpine sh -c '\''
  apk add --quiet sqlite > /dev/null
  sqlite3 /data/webui.db "
    UPDATE config
    SET data = json_set(data, '\''"\$"'\''.rag.web.search.bypass_embedding_and_retrieval, json('\''"false"'\''));"
'\''

sudo docker start open-webui
```

(Or you can edit through the OWUI UI — Admin Panel → Settings → Web Search — and just hope the in-memory state matches what you saved.)

### What this setup delivers (with all four layers in place)

- ✅ Native ROCm acceleration on gfx1103 (~3-5× faster than CPU)
- ✅ Measured 52/52 successful web-search chats in a real-world session
- ✅ Stable for typical chat workloads, OWUI web-search use, and Matrix-bot-style API traffic
- ✅ ~13-14 tok/s short context, ~6-8 tok/s at 14k context

### What it does not deliver

- ❌ A 100% guarantee — the MES bug is reduced not eliminated by kernel `-15`; consumers should still expect occasional retries on long-context heavy workloads.
- ❌ Production-grade reliability for unattended workloads with no retry logic.
- ❌ Stability at high context (>16k) or default batch size — the MES bug returns at scale.

### Alternative: Vulkan backend

If for some reason you can't run a `-15`-or-newer kernel, or if you need fully unattended reliability, Ollama supports Vulkan via Mesa RADV instead of ROCm. RADV has very mature gfx1103 support and is reportedly crash-free even on older kernels, at the cost of roughly 30-40% throughput vs. ROCm. The ROCm path in this README is the recommended one with current kernels, but Vulkan is a real escape hatch.

## Requirements

- Ubuntu 26.04 LTS (other Ubuntu/Debian versions probably work but untested)
- **Linux kernel `7.0.0-15-generic` or newer** — the HWE kernel that ships with `linux-image-generic-hwe-26.04`. Older kernels (`-14` and earlier) have a gfx11 MES bug that causes ~14% of inference requests to crash with a GPU reset. The setup script checks for this and will refuse to proceed if your running kernel is too old.
- AMD Phoenix-family CPU (Ryzen 7040 / 8040 / etc.) with Radeon 780M iGPU
- ROCm 7.1 system packages from Ubuntu's universe repo (the script installs these)
- A few GB of disk for the build artifacts and Fedora RPM
- 10–30 minutes of CPU time for the C++/HIP build

If you've got a *different* AMD GPU (gfx1100 / gfx1101 / gfx1102 / gfx1151), this isn't strictly necessary for you — Ubuntu/AMD probably ship working kernels for your arch. Use this only if you actually need gfx1103.

## Quick start

```bash
git clone https://github.com/johnsonfarmsus/ollama-rocm-gfx1103-ubuntu.git
cd ollama-rocm-gfx1103-ubuntu

# Single-GPU system (only the Phoenix iGPU):
./setup.sh

# Multi-GPU system — pin to the iGPU's physical HIP index
# (check `rocminfo` to find which index is the 780M; typically 1 if you also have a dGPU):
ROCR_PIN_DEVICE=1 ./setup.sh
```

The script is idempotent — re-running is safe. In particular, **re-run after every `apt upgrade librocblas5`**, because the upgrade overwrites `/usr/lib/x86_64-linux-gnu/rocblas/5.1.0/library/` with Ubuntu's gfx1103-less package contents.

## Verification

After the script finishes, check that Ollama actually sees the iGPU as ROCm:

```bash
sudo journalctl -u ollama -n 30 | grep "inference compute"
```

You want to see `library=ROCm compute=gfx1103`, not `library=cpu`. If you see CPU, something failed silently — read the surrounding journal output for the actual error.

Then pull a model and inference:

```bash
ollama pull gemma4:e2b   # or any model you like
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model": "gemma4:e2b",
  "stream": false,
  "messages": [{"role": "user", "content": "What is the capital of France?"}]
}' | python3 -m json.tool
```

You should get an answer in a few seconds. Compute `eval_count / eval_duration * 1e9` from the response for tokens/sec.

## What's in here

- [`setup.sh`](setup.sh) — the automation script. Idempotent.
- [`patches/`](patches/) — three patches to apply against `ml/device.go` in `ollama-for-amd`:
  - [`01-sort-by-free-memory.patch`](patches/01-sort-by-free-memory.patch) — fixes the device-selection sort so the iGPU isn't downranked relative to a smaller dGPU. This is upstreamable as a real correctness fix.
  - [`02-skip-init-validation.patch`](patches/02-skip-init-validation.patch) — skips Ollama's "deep init" GPU validation that would crash even working iGPUs.
  - [`03-respect-parent-rocr-visible-devices.patch`](patches/03-respect-parent-rocr-visible-devices.patch) — keeps the runner subprocess on the GPU pinned via systemd, even after Ollama's internal scheduler runs.
- [`override.conf`](override.conf) — systemd drop-in template installed at `/etc/systemd/system/ollama.service.d/override.conf`.

## What's not in here (and why)

- **Patched `rocBLAS` source.** Building rocBLAS from source with `-DAMDGPU_TARGETS=gfx1103` is an alternative to grabbing Fedora's prebuilt kernels. It takes hours and produces functionally the same result. The script uses Fedora's binaries because they work, they're already built, and the .hsaco files are pure GPU bytecode that doesn't care about your host distribution.
- **A binary release.** The `ollama` binary and `libggml-hip.so` need to be built on a machine with ROCm 7.x dev headers matching your runtime. A pre-built binary would have to match a specific runtime version; instead, the script rebuilds each time you run it.

## Caveats / shelf life

This whole approach is held together by a specific set of gaps in the ecosystem. Some of them will close in time:

- When Ubuntu eventually ships gfx1103 kernels in their `librocblas5` package (likely in 27.04 or whenever ROCm 7.x integrates Phoenix), step 6 of the script becomes unnecessary.
- When upstream Ollama integrates the AMD-tuned code (or fixes the sort bias separately), the patches in this repo may stop applying cleanly. If `git apply --check` fails, the patches need to be re-derived against the new upstream.
- An `apt upgrade` of `librocblas5` will overwrite the kernel files. Re-run `setup.sh` afterward.
- **Kernel ABI must match runtime ABI.** Earlier versions of this script used Fedora 43's rocBLAS 6.4 kernels with Ubuntu's rocBLAS 7.1 runtime — this looked like it worked but caused frequent crashes. The current script uses Fedora 44's matching rocBLAS 7.1.1 build. When you update your system's rocBLAS (e.g., to 7.2 in a future Ubuntu release), update the Fedora RPM URL in `setup.sh` to a Fedora build at the matching version.
- **Linux kernel version matters.** The amdgpu MES `REMOVE_QUEUE` fix that drops crash rate from ~14% to 0% landed in Linux ~6.11–6.13 and is in Ubuntu's HWE kernel `7.0.0-15`. If a future Ubuntu LTS ships an older kernel by default, you'll want to install the HWE option to pick this up. The script checks the running kernel and refuses to proceed if it's too old.

None of these are imminent as of May 2026.

## Contributing

If you find the patches don't apply against a newer ollama-for-amd checkout, or the Fedora RPM URL changes, open an issue or PR. The script tries to be defensive (refuses to proceed if it can't find expected files) but the moving parts mean breakage is inevitable eventually.

The first patch — sort by free memory — has been submitted upstream to Ollama proper. Track at [#TODO-pr-link].

## Sources & references

- [ROCm/rocBLAS issue #1536 — gfx1103 (Phoenix) support request](https://github.com/ROCm/rocBLAS/issues/1536) (still open as of May 2026)
- [Fedora rocblas package](https://packages.fedoraproject.org/pkgs/rocblas/) — where the gfx1103 kernels come from
- [likelovewant/ollama-for-amd](https://github.com/likelovewant/ollama-for-amd) — the Ollama fork with broader AMD GPU support
- [likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU) — Windows-focused but useful background
- [Blog post writeup](https://ataary.com/ubuntu-linux-with-ollama-rocm-on-amd-ryzen-780m-igpu/) — the full story of why each piece is necessary

## License

The patches and setup script in this repo are released under the MIT license; see [LICENSE](LICENSE). The upstream projects (Ollama, ollama-for-amd, rocBLAS, Fedora rocblas package) retain their own licenses.
