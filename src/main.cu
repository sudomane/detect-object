#include <iostream>
#include <stdio.h>
#include <stddef.h>

#include <cuda.h>
#include <png.h>

__global__ void my_kernel()
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Hello from the GPU! (Thread n %d)\n", i);
}

int main()
{
    printf("Hello world!\n");

    my_kernel<<<2, 2>>>();
    cudaDeviceSynchronize();
}