# torch-harmonics MUSA integration

The Moore Threads port adds MUSA extension sources for the attention and DISCO
operators while retaining the upstream Python interfaces and CUDA backend.
Build integration selects the MUSA compiler and runtime through the existing
Python extension configuration. The MUSA SDK is an external prerequisite; the
repository does not bundle that toolchain.
