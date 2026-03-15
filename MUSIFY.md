# MUSIFY — torch-harmonics

| Field | Value |
|-------|-------|
| Project | torch-harmonics |
| Source repo | https://github.com/NVIDIA/torch-harmonics |
| Migration branch | musa |
| Artifact type | Ext (Python Extension) |
| Build system | setup.py with CUDAExtension |
| Test infrastructure | pytest (tests/), 11 test files |
| MUSA SDK version | 4.3.5 |
| torch_musa version | v2.7.1 |
| MT GPU type | S5000 |
| Pipeline start | 2026-03-15 01:42:29 UTC |
| Pipeline end | |
| Elapsed | |
| Last updated | 2026-03-15 01:46 |
| Director | xiaofeng |
| Status | in-progress |

## Migration Plan

Migrate torch-harmonics CUDA kernels (disco + attention modules) and Python layer to MUSA. Two CUDA extension modules: `disco._C` and `attention._C`. Each has dedicated `.cu`/`.cuh` files for forward/backward kernels. Python layer uses `torch.cuda.*` APIs extensively in tests, examples, and library code.

## CUDA Dependencies

None -- all dependencies are MUSA-compatible or CUDA-free. The project uses CUB (bundled with CUDA Toolkit, mapped to MUSA equivalent) and ATen/c10 CUDA headers (mapped via torch_musa). No third-party CUDA packages need musification.

## Pipeline Progress

### torch-harmonics (Main)

| Stage | Status | Started | Completed | Notes |
|-------|--------|---------|-----------|-------|
| 0. sdk-version | done | 2026-03-15 01:42 | 2026-03-15 01:44 | v2.7.1 / SDK 4.3.5 |
| 1a. reader (code analysis) | done | 2026-03-15 01:44 | 2026-03-15 01:46 | 8 .cu, 2 .cuh, ~25 Python files |
| 1b. lookuper (document-site) | skipped | | | Not needed for this migration |
| 1c. lookuper (docker setup) | done | 2026-03-15 01:44 | 2026-03-15 01:45 | Container reused: musify_container |
| 2. api-mapping | in-progress | 2026-03-15 01:46 | | |
| 3. musa-adapt | pending | | | |

#### Stage Details

**Stage 0 -- SDK Version Resolution (Core):**
- torch_musa version: v2.7.1
- MUSA SDK version: 4.3.5
- Docker image: 4.3.5_kuae2.1_20260119_torch2.7.1_ubuntu

**Stage 1 -- Parallel Preparation:**

*1a. Reader (code analysis):*
- Project type: Ext (Python Extension with CUDAExtension)
- CUDA files: 8 .cu (disco: 2 fwd/bwd, attention: 5 fwd/bwd/utils/interface), 2 .cuh (disco_cuda.cuh, attention_cuda.cuh, attention_cuda_utils.cuh, cudamacro.h)
- Python files with torch.cuda: ~25 files (tests, examples, library code)
- Minimum CUDA version: 11.0+ (uses `__CUDA_ARCH__` with SM 90 guards in attention_cuda_bwd.cu)
- CUDA ecosystem: CUB (cub::WarpReduce), ATen/c10 CUDA headers
- Two CUDAExtension modules: `torch_harmonics.disco._C`, `torch_harmonics.attention._C`

*1c. Lookuper (docker setup):*
- Container name: musify_container
- GPU visible: yes
- MUSA version: 4.3.5
- torch_musa version: 2.7.1+5ee0a64

## Issues

| # | Team | Stage | Severity | Summary | Status | Resolution |
|---|------|-------|----------|---------|--------|------------|

## Knowledge Base

## Final Report
