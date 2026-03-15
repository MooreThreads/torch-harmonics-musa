#!/usr/bin/env python3
"""
SimplePorting script for torch-harmonics CUDA-to-MUSA migration.
Run inside the container:
    python3 simple_porting_torch_harmonics.py
"""
import os
import shutil
from torch_musa.utils.simple_porting import SimplePorting

PROJECT_DIR = "/workspace/torch-harmonics"

# Custom mapping rules (project-specific, extend defaults)
MAPPING_RULES = {
    # ATen CUDA headers -> torch_musa equivalents (quoted includes)
    '#include <ATen/cuda/CUDAContext.h>': '#include "torch_musa/csrc/aten/musa/MUSAContext.h"',
    '#include <ATen/cuda/CUDAUtils.h>': '#include "torch_musa/csrc/aten/musa/MUSAContext.h"',
    '#include <ATen/cuda/detail/TensorInfo.cuh>': '#include <ATen/detail/TensorInfo.h>',
    '#include <ATen/cuda/detail/KernelUtils.h>': '#include <ATen/detail/KernelUtils.h>',
    '#include <ATen/cuda/detail/IndexUtils.cuh>': '#include <ATen/detail/IndexUtils.h>',

    # c10/cuda headers -> torch_musa equivalents
    '#include <c10/cuda/CUDAStream.h>': '#include "torch_musa/csrc/core/MUSAStream.h"',

    # Namespace fixes
    "at::cuda::getCurrentCUDAStream()": "at::musa::getCurrentMUSAStream()",
    "at::cuda::": "at::musa::",
    "c10::cuda::": "c10::musa::",

    # Device type checks
    "torch::kCUDA": "torch::kPrivateUse1",
    ".is_cuda()": ".is_privateuseone()",

    # Compiler macros
    "__NVCC__": "__MUSACC__",

    # Local header renames (.cuh -> .muh)
    '"disco_cuda.cuh"': '"disco_cuda.muh"',
    '"attention_cuda.cuh"': '"attention_cuda.muh"',
    '"attention_cuda_utils.cuh"': '"attention_cuda_utils.muh"',
    '"cudamacro.h"': '"musamacro.h"',

    # cudamacro.h content replacements (will be applied to the copied file)
    "cudaError_t": "musaError_t",
    "cudaSuccess": "musaSuccess",
    "cudaGetErrorString": "musaGetErrorString",
    "cudaGetLastError": "musaGetLastError",
    "Cuda error": "MUSA error",
}

def clean_existing_musa_dirs():
    """Remove existing _musa directories before regeneration."""
    for subdir in ["torch_harmonics/disco/csrc_musa", "torch_harmonics/attention/csrc_musa"]:
        path = os.path.join(PROJECT_DIR, subdir)
        if os.path.exists(path):
            print(f"Removing: {path}")
            shutil.rmtree(path)

def port_directories():
    """Port each CUDA directory using SimplePorting."""
    dirs_to_port = [
        # (cuda_dir, ignore_dirs)
        (os.path.join(PROJECT_DIR, "torch_harmonics/disco/csrc"), []),
        (os.path.join(PROJECT_DIR, "torch_harmonics/attention/csrc"), []),
    ]
    for cuda_dir, ignore_dirs in dirs_to_port:
        if os.path.exists(cuda_dir):
            print(f"Porting {cuda_dir}")
            SimplePorting(
                cuda_dir_path=cuda_dir,
                ignore_dir_paths=ignore_dirs,
                mapping_rule=MAPPING_RULES,
                drop_default_mapping=False,
            ).run()
            print(f"  -> Created {cuda_dir}_musa/")
        else:
            print(f"WARNING: {cuda_dir} does not exist, skipping")

if __name__ == "__main__":
    clean_existing_musa_dirs()
    port_directories()
    print("SimplePorting complete.")
