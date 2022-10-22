#include <stdio.h>
#include <spdlog/spdlog.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include "image.hpp"

// Unit test functions
void test_grayscale(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int d_img_pitch);
void test_conv_2D(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int d_img_pitch);

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: ./main [path_to_image]\n");
        return -1;
    }

    cudaError_t rc = cudaSuccess;

    u_char* d_img;
    size_t d_img_pitch;

    u_char* h_img;
    int width, height, n_channels;

    // Allocate image on host, for CPU specific tasks
    {
        h_img = stbi_load(argv[1], &width, &height, &n_channels, 0);
        if (h_img == nullptr)
        {
            spdlog::error("Could not find image {}", argv[1]);
            return -1;
        }
        spdlog::info("[CPU] Loaded image {} | {}x{}x{}",
                     argv[1], width, height, n_channels);
    }

    // Allocate image on device as well, for GPU specific tasks.
    {
        rc = cudaMallocPitch(&d_img, &d_img_pitch,
                             width * n_channels * sizeof(u_char), height);
        if (rc)
        {
            spdlog::error("Failed device image allocation. Error code {}", rc);
            return -1;
        }
        rc = cudaMemcpy2D(d_img, d_img_pitch, h_img, width * n_channels,
                          width * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
        {
            spdlog::error("Failed to copy image to device. Error code {}", rc);
            return -1;
        }
        
        spdlog::info("[GPU] Loaded image {} | {}x{}x{}",
                     argv[1], width, height, n_channels);
    }

    test_grayscale(h_img, d_img, width, height, n_channels, d_img_pitch);
    test_conv_2D(h_img, d_img, width, height, n_channels, d_img_pitch);

    free(h_img);
    cudaFree(d_img);
    
    return 0;
}

void test_grayscale(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int d_img_pitch)
{       
    cudaError_t rc  = cudaSuccess;

    int block_size  = 32;
    int w           = std::ceil((float)width / block_size);
    int h           = std::ceil((float)height / block_size);
    
    dim3 dimGrid(w, h);
    dim3 dimBlock(block_size, block_size);

    // Buffers to store grayscale results on.
    u_char* h_img_gray_CPU = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    u_char* h_img_gray_GPU = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    
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

    // CPU Grayscale test
    {
        CPU::to_grayscale(h_img, h_img_gray_CPU, width, height, n_channels);
        stbi_write_jpg("../out_gray_CPU.jpeg", width, height, 1, h_img_gray_CPU, width);
        spdlog::info("[CPU] Successfully converted image to grayscale.");
    }

    // GPU Grayscale test
    {
        GPU::to_grayscale<<<dimGrid, dimBlock>>>(d_img, d_img_gray, width, height,
                                                 d_img_pitch, d_img_gray_pitch, n_channels);
        rc = cudaDeviceSynchronize();
        if (rc)
        {
            spdlog::error("Kernel failed. Error code: {}", rc);
            return;   
        }

        rc = cudaMemcpy2D(h_img_gray_GPU, width, d_img_gray, d_img_gray_pitch,
                          width * sizeof(char), height, cudaMemcpyDeviceToHost);
        if (rc)
        {
            spdlog::error("Failed to copy image from device to host. Error code: {}", rc);
            return;  
        }

        stbi_write_jpg("../out_gray_GPU.jpeg", width, height, 1, h_img_gray_GPU, width);
        spdlog::info("[GPU] Successfully converted image to grayscale.");
    }
    
    // Free buffers
    free(h_img_gray_CPU);
    free(h_img_gray_GPU);
    cudaFree(d_img_gray);
}

void test_conv_2D(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int d_img_pitch)
{
    u_char* h_img_gray;
    u_char* h_img_conv;
    
    // Allocate host images for grayscale and conv
    {
        h_img_gray = static_cast<u_char*>(malloc(width * height * sizeof(u_char)));
        h_img_conv = static_cast<u_char*>(malloc(width * height * sizeof(u_char)));
       
        if (h_img_gray == nullptr)
        {
            spdlog::error("Could not allocate memory for grayscale image.");
            return;
        }

        if (h_img_conv == nullptr)
        {
            spdlog::error("Could not allocate memory for convoluted image.");
            return;
        }
    }
    
    CPU::to_grayscale(h_img, h_img_gray, width, height, n_channels);
    CPU::conv_2D(h_img_gray, h_img_conv, width, height);

    stbi_write_jpg("../out_conv_CPU.jpeg", width, height, 1, h_img_conv, width);
    spdlog::info("[CPU] Successfully applied 2D convolution on image.");

    free(h_img_gray);
    free(h_img_conv);
}