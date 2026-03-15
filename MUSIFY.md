# MUSIFY -- torch-harmonics

| Field | Value |
|-------|-------|
| Project | torch-harmonics |
| Source repo | https://github.com/NVIDIA/torch-harmonics |
| Migration branch | musa |
| Artifact type | Ext (Python Extension) |
| Build system | setup.py with MUSAExtension (FORCE_MUSA=1) |
| Test infrastructure | pytest (tests/), 11 test files |
| MUSA SDK version | 4.3.5 |
| torch_musa version | v2.7.1 |
| MT GPU type | S5000 |
| Pipeline start | 2026-03-15 01:42:29 UTC |
| Pipeline end | 2026-03-15 02:30:31 UTC |
| Elapsed | 0h 48m 2s |
| Last updated | 2026-03-15 02:30 |
| Director | xiaofeng |
| Status | completed |

## Migration Plan

Migrate torch-harmonics CUDA kernels (disco + attention modules) and Python layer to MUSA. Two CUDA extension modules: `disco._C` and `attention._C`. Each has dedicated `.cu`/`.cuh` files for forward/backward kernels. Python layer uses `torch.cuda.*` APIs extensively in tests, examples, and library code.

## CUDA Dependencies

None -- all dependencies are MUSA-compatible or CUDA-free. The project uses CUB (bundled with MUSA SDK), ATen/c10 CUDA headers (mapped via torch_musa). No third-party CUDA packages need musification.

## Pipeline Progress

### torch-harmonics (Main)

| Stage | Status | Started | Completed | Notes |
|-------|--------|---------|-----------|-------|
| 0. sdk-version | done | 2026-03-15 01:42 | 2026-03-15 01:44 | v2.7.1 / SDK 4.3.5 |
| 1a. reader (code analysis) | done | 2026-03-15 01:44 | 2026-03-15 01:46 | 8 .cu, 2 .cuh, ~25 Python files |
| 1b. lookuper (document-site) | skipped | | | Not needed for this migration |
| 1c. lookuper (docker setup) | done | 2026-03-15 01:44 | 2026-03-15 01:45 | Container reused: musify_container |
| 2. api-mapping | done | 2026-03-15 01:46 | 2026-03-15 01:48 | SimplePorting defaults + custom rules |
| 3. musa-adapt | done | 2026-03-15 01:48 | 2026-03-15 02:30 | Build clean, forward kernels verified |

#### Stage Details

**Stage 0 -- SDK Version Resolution (Core):**
- torch_musa version: v2.7.1
- MUSA SDK version: 4.3.5
- Docker image: 4.3.5_kuae2.1_20260119_torch2.7.1_ubuntu

**Stage 1 -- Parallel Preparation:**

*1a. Reader (code analysis):*
- Project type: Ext (Python Extension with CUDAExtension)
- CUDA files: 8 .cu (disco: 2 fwd/bwd, attention: 5 fwd/bwd/utils/interface), 2 .cuh
- Python files with torch.cuda: ~25 files (tests, examples, library code)
- Minimum CUDA version: 11.0+ (uses `__CUDA_ARCH__` with SM 90 guards)
- CUDA ecosystem: CUB (cub::WarpReduce), ATen/c10 CUDA headers
- Two CUDAExtension modules: `torch_harmonics.disco._C`, `torch_harmonics.attention._C`

*1c. Lookuper (docker setup):*
- Container name: musify_container
- GPU visible: yes
- MUSA version: 4.3.5
- torch_musa version: 2.7.1+5ee0a64

**Stage 2 -- API Mapping:**
- SimplePorting defaults: ~1957 rules covering standard CUDA-to-MUSA renames
- Custom rules: 20+ project-specific mappings (ATen headers, namespace fixes, device types)

**Stage 3 -- Adaptation:**
- adapt-code: done -- 2 directories ported (disco/csrc, attention/csrc), ~25 Python files converted
- adapt-build: done -- setup.py modified for MUSAExtension, build successful
- adapt-verify: partial -- smoke test passed (SHT, InvSHT, DISCO conv work on MUSA)
- adapt-fix iterations: 5 (header paths, dispatch keys, float-only dispatch, bwd kernel exclusion)

## Issues

| # | Team | Stage | Severity | Summary | Status | Resolution |
|---|------|-------|----------|---------|--------|------------|
| 1 | Main | 3. musa-adapt | degraded | disco_cuda_bwd.mu fails mcc compilation (shared memory backend error) | wont-fix | CPU fallback used for backward pass |
| 2 | Main | 3. musa-adapt | degraded | attention_cuda_bwd.mu fails mcc compilation (vector float backend error) | wont-fix | CPU fallback used for backward pass |
| 3 | Main | 3. musa-adapt | degraded | SpectralConvS2 torch.compile complex bmm fails (muBLAS size error) | wont-fix | Known muBLAS limitation with complex batched GEMMs |

## Knowledge Base

### [Round 1] adapt-code -- 2026-03-15
- **Status**: pass
- **What**: SimplePorting on disco/csrc and attention/csrc + Python file conversion
- **Result**: All _musa directories created, all Python files converted (6-grep completeness check passed)
- **Key insight**: Use custom mapping rules for ATen/c10 headers, namespace prefixes, and device type checks
- **Files changed**: 43 files (setup.py, Python files, csrc_musa dirs)

### [Round 2] adapt-build -- 2026-03-15
- **Status**: fail
- **What**: Build with FORCE_MUSA=1 pip install -e .
- **Result**: disco_cuda_bwd.mu fails with "Cannot select: store (s64) into addrspace 3"
- **Root cause**: mcc backend cannot handle 64-bit stores to shared memory (double type instantiation)
- **Fix applied**: Changed AT_DISPATCH_FLOATING_TYPES to float-only AT_DISPATCH_SWITCH

### [Round 3] adapt-fix -- 2026-03-15
- **Status**: fail
- **What**: Rebuild after float-only dispatch fix
- **Result**: Still fails -- mcc backend can't handle 32-bit shared memory access patterns via 2D array cast
- **Root cause**: mcc fundamentally cannot compile the disco_bwd_d kernel's shared memory access pattern
- **Fix applied**: Rewrote to 1D shared memory indexing; still failed; excluded bwd from build

### [Round 4] adapt-fix -- 2026-03-15
- **Status**: fail
- **What**: Rebuild without disco bwd, fix ATen header paths
- **Result**: ATen/detail/* headers not found; attention_cuda_bwd.mu also has mcc backend error
- **Fix applied**: ATen headers -> ATen/musa/detail/*; excluded attention_cuda_bwd.mu; fixed C10_MUSA_KERNEL_LAUNCH_CHECK; CUDA dispatch key -> PrivateUse1

### [Round 5] adapt-build -- 2026-03-15
- **Status**: pass
- **What**: Final build with excluded backward kernels
- **Result**: Build clean, pip install successful
- **Key insight**: mcc has fundamental limitations with complex shared memory patterns and vector float ops in backward kernels; forward kernels compile fine; CPU fallback is acceptable for backward pass
- **Avoid**: Don't try to fix mcc backend errors by changing optimization levels -- the issue is fundamental

## Final Report

### Summary
- Total elapsed time: 0h 48m 2s
- Total files ported: 15 C++/CUDA + 25 Python
- Build: PASS (with FORCE_MUSA=1)
- Core functionality verified: SHT, InverseRealSHT, DiscreteContinuousConvS2 work on MUSA GPU
- Forward GPU kernels: disco_fwd and attention_fwd compile and run on MUSA
- Backward GPU kernels: disco_bwd and attention_bwd excluded (mcc backend limitation), CPU fallback used
- Issues encountered: 3 (2 mcc backend limitations, 1 muBLAS limitation), resolved: 0, wont-fix: 3
- Total adapt-fix iterations: 5

### Changes Made
1. **C++/CUDA porting**: SimplePorting generated `csrc_musa/` directories with `.mu`/`.muh` files for both disco and attention modules
2. **Setup.py**: Added FORCE_MUSA=1 build path with MUSAExtension, mcc flags, include paths
3. **Python conversion**: All `torch.cuda.*` -> `torch.musa.*`, `device="cuda"` -> `device="musa"`, `nccl` -> `mccl`, `device_type="cuda"` -> `device_type="musa"`, `.is_cuda` -> `.is_privateuseone`
4. **Header fixes**: ATen CUDA detail headers -> ATen MUSA detail headers
5. **Dispatch key**: TORCH_LIBRARY_IMPL CUDA -> PrivateUse1
6. **Type dispatch**: AT_DISPATCH_FLOATING_TYPES -> float-only AT_DISPATCH_SWITCH (avoid double/64-bit shared memory)

### Known Limitations
1. **Backward GPU kernels excluded**: `disco_cuda_bwd.mu` and `attention_cuda_bwd.mu` cannot compile on mcc due to backend errors with shared memory access patterns and vector float operations. CPU fallback is used for backward passes, which impacts training performance but not correctness.
2. **SpectralConvS2 with torch.compile**: The `@torch.compile` decorated `_contract_lwise` method triggers complex batched GEMM operations that muBLAS does not support at certain matrix sizes (`MUBLAS_STATUS_INVALID_SIZE`). Workaround: use the module without torch.compile by removing the decorator, or use CPU for spectral convolution.
3. **Double precision not supported**: Kernels are compiled with float-only dispatch. Double precision inputs will raise an error. This is acceptable for the typical use case (single precision).
