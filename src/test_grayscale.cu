#include "test.hpp"

#include <spdlog/spdlog.h>

#include "image.hpp"
#include "img_operations.hpp"

void test_grayscale_CPU(unsigned char* h_img, int width, int height, int n_channels)
{
    u_char* h_img_gray = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    
    CPU::to_grayscale(h_img, h_img_gray, width, height, n_channels);
    write_image("../out_gray_CPU.jpeg", width, height, h_img_gray);
    
    spdlog::info("[CPU] Successfully converted image to grayscale.");

    free(h_img_gray);
}


void test_grayscale_GPU(unsigned char* d_img, int width, int height, int n_channels, int pitch)
{
    cudaError_t rc  = cudaSuccess;

    int block_size  = 32;
    int w           = std::ceil((float)width / block_size);
    int h           = std::ceil((float)height / block_size);
    
    dim3 dimGrid(w, h);
    dim3 dimBlock(block_size, block_size);

    u_char* h_img_gray = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));

    // Initialize buffer to store intermediate device operations for grayscale image
    // d_img ---> d_img_gray --[Memcpy2D]--> h_img_gray_GPU
    // <------ device ----->                 <--- host --->
    u_char* d_img_gray;
    size_t d_img_gray_pitch;
    rc = cudaMallocPitch(&d_img_gray, &d_img_gray_pitch, width * sizeof(u_char), height);
    if (rc)
    {
        spdlog::error("Failed device image allocation. Error code: {}", rc);
        return;
    }

    // GPU Grayscale test
    {
        GPU::to_grayscale<<<dimGrid, dimBlock>>>(d_img, d_img_gray, width, height,
                                                 pitch, d_img_gray_pitch, n_channels);
        rc = cudaDeviceSynchronize();
        if (rc)
        {
            spdlog::error("Kernel failed. Error code: {}", rc);
            return;   
        }

        rc = cudaMemcpy2D(h_img_gray, width, d_img_gray, d_img_gray_pitch,
                          width * sizeof(char), height, cudaMemcpyDeviceToHost);
        if (rc)
        {
            spdlog::error("Failed to copy image from device to host. Error code: {}", rc);
            return;  
        }

        write_image("../out_gray_GPU.jpeg", width, height, h_img_gray);
        spdlog::info("[GPU] Successfully converted image to grayscale.");
    }

    free(h_img_gray);
    cudaFree(d_img_gray);
}
