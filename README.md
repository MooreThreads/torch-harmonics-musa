# torch-harmonics-musa

`torch-harmonics-musa` is the Moore Threads MUSA maintenance line of the
upstream [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) project.
It provides differentiable spherical-harmonic transforms, DISCO convolutions,
and attention operators for PyTorch workloads.

## Source and version

- Upstream project: NVIDIA torch-harmonics
- Inherited version: `v0.8.1`
- Upstream baseline: `8b9f5230e07c64a9e7fdc4dba937b2f4ffd59157`
- Maintainer: Moore Threads

The MUSA line adds MUSA extension sources and keeps the upstream Python API,
CUDA sources, tests, and project layout available. Backend selection remains
explicit so downstream users can choose CUDA or MUSA as appropriate.

## Requirements and build

Use a supported Python and PyTorch installation together with the MUSA SDK and
its compiler/runtime. A typical editable installation is:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Select the MUSA build option documented by `setup.py` when building the MUSA
extensions. The exact SDK and PyTorch versions are supplied by the deployment
environment.

## Layout and contribution

Python APIs live under `torch_harmonics/`; backend extension sources and tests
are kept alongside the inherited project structure. Contributions should keep
MUSA and CUDA paths separate, preserve upstream attribution, and include a
focused description of the affected backend.

This repository is maintained by Moore Threads from the upstream source line.
See [LICENSE](LICENSE) for the applicable declaration.
