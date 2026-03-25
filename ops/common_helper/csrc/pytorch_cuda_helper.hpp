#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>

#include "common_cuda_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;
using at::BFloat16;
using pbhalf = at::BFloat16;

#define __PHALF(x) (x)
#define __PBHALF(x) (x)

#endif  // PYTORCH_CUDA_HELPER
