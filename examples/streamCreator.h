#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef streamCreator_h
#define streamCreator_h
cudaStream_t createStreamWithFlags();
#endif

