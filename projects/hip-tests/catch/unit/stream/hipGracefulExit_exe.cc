/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Regression test for: "Don't wait on command completion if worker thread
// is destroyed" (rocm-systems PR #3790).
//
// This program queues async GPU work and exits without explicit
// synchronization or cleanup. During process exit, the atexit path
// (__hipUnregisterFatBinary -> SyncAllStreams -> HostQueue::finish) must
// handle dead worker threads gracefully.
//
// On Windows (AMD_DIRECT_DISPATCH=false by default), ExitProcess kills
// all non-main threads before running atexit handlers, so the worker
// thread is guaranteed dead when finish() is called.
//
// Before the fix: hangs indefinitely during process exit.
// After the fix: exits cleanly.

#include <hip/hip_runtime.h>

__global__ void add_one(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] += 1.0f;
}

int main() {
  float* d_data;
  if (hipMalloc(&d_data, 1024 * sizeof(float)) != hipSuccess) {
    return 1;
  }
  if (hipMemset(d_data, 0, 1024 * sizeof(float)) != hipSuccess) {
    return 1;
  }

  // Launch async work — intentionally no hipStreamSynchronize, no hipFree,
  // no hipStreamDestroy. Let atexit handlers clean up, which is what
  // frameworks like PyTorch do during interpreter shutdown.
  add_one<<<4, 256>>>(d_data, 1024);

  return 0;
}
