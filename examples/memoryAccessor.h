#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef memoryAccessor_h
#define memoryAccessor_h
void cuSetDeviceFlags();
void cuMallocManaged(void** h_img, int r, int c, int channel);
void cuMalloc(void** h_img, int r, int c);
void cuDeviceSynchronize();
void cuFree(void* mem);
#endif
