#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
void cuSetDeviceFlags(){
	cudaSetDeviceFlags(cudaDeviceMapHost);
}
void cuMallocManaged(void** h_img, int r, int c, int channel){

	cudaMallocManaged(h_img,sizeof(unsigned char)*r*c * channel);

}

void cuMalloc(void** h_img, int r, int c){
	cudaMalloc(h_img, sizeof(float)*r*c);
}

void cuDeviceSynchronize(){
	cudaDeviceSynchronize();
}

void cuFree(void* mem){
	cudaFree(mem);
}
