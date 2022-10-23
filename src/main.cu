#include <stdio.h>
#include <spdlog/spdlog.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include "image.hpp"

// Unit test functions
void test_grayscale(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int pitch);
void test_conv_2D(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int pitch);

// Separated into 2 different functions, too messy otherwise!
void test_diff_CPU(u_char* h_img_1, u_char* h_img_2, int width, int height, int n_channels);
void test_diff_GPU(u_char* d_img_1, u_char* d_img_2, int width, int height, size_t pitch, int n_channels);

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        printf("Usage: ./main [image 1] [image 2]\n");
        return -1;
    }

    cudaError_t rc = cudaSuccess;

    u_char* d_img_1;
    u_char* d_img_2;
    size_t pitch;

    u_char* h_img_1;
    u_char* h_img_2;
    int width, height, n_channels;
    int width_, height_, n_channels_;

    // Allocate image on host, for CPU specific tasks
    {
        h_img_1 = stbi_load(argv[1], &width, &height, &n_channels, 0);
        if (h_img_1 == nullptr)
        {
            spdlog::error("Could not find image {}", argv[1]);
            return -1;
        }

        h_img_2 = stbi_load(argv[2], &width_, &height_, &n_channels_, 0);
        if (h_img_2 == nullptr)
        {
            spdlog::error("Could not find image {}", argv[1]);
            return -1;
        }

        if (width != width_ || height != height_ || n_channels != n_channels_)
        {
            spdlog::error("Image dimensions mismatch! {}x{}x{} against {}x{}x{}",
                          width, height, n_channels, width_, height_, n_channels_);
            return -1;
        }
        
        spdlog::info("[CPU] Loaded image {} {} | {}x{}x{}",
                     argv[1], argv[2], width, height, n_channels);
    }

    // Allocate image on device as well, for GPU specific tasks.
    // Since both images are of exact same dimensions, it's ok to overwrite the pitch.
    {
        rc = cudaMallocPitch(&d_img_1, &pitch,
                             width * n_channels * sizeof(u_char), height);
        if (rc)
        {
            spdlog::error("Failed device image (1) allocation. Error code {}", rc);
            return -1;
        }
        rc = cudaMemcpy2D(d_img_1, pitch, h_img_1, width * n_channels,
                          width * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
        {
            spdlog::error("Failed to copy image (1) to device. Error code {}", rc);
            return -1;
        }
        
        rc = cudaMallocPitch(&d_img_2, &pitch,
                             width * n_channels * sizeof(u_char), height);
        if (rc)
        {
            spdlog::error("Failed device image (2) allocation. Error code {}", rc);
            return -1;
        }
        rc = cudaMemcpy2D(d_img_2, pitch, h_img_2, width * n_channels,
                          width * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
        {
            spdlog::error("Failed to copy image (2) to device. Error code {}", rc);
            return -1;
        }

        spdlog::info("[GPU] Loaded images {} {} | {}x{}x{}",
                     argv[1], argv[2], width, height, n_channels);
    }

    // Running the tests
    
    test_grayscale(h_img_1, d_img_1, width, height, n_channels, pitch);
    test_conv_2D(h_img_1, d_img_1, width, height, n_channels, pitch);
    test_diff_CPU(h_img_1, h_img_2, width, height, n_channels);

    free(h_img_1);
    free(h_img_2);
    cudaFree(d_img_1);
    cudaFree(d_img_2);
    
    return 0;
}

void test_grayscale(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int pitch)
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
                                                 pitch, d_img_gray_pitch, n_channels);
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

void test_conv_2D(u_char* h_img, u_char* d_img, int width, int height, int n_channels, int pitch)
{
    u_char* h_img_gray;
    u_char* h_img_conv;
    
    u_char* d_img_conv;
    // CPU Convolution test
    {
        h_img_gray = static_cast<u_char*>(malloc(width * height * sizeof(u_char)));
        h_img_conv = static_cast<u_char*>(malloc(width * height * sizeof(u_char)));

        CPU::to_grayscale(h_img, h_img_gray, width, height, n_channels);
        CPU::conv_2D(h_img_gray, h_img_conv, width, height);

        stbi_write_jpg("../out_conv_CPU.jpeg", width, height, 1, h_img_conv, width);
        spdlog::info("[CPU] Successfully applied 2D convolution on image.");
    }

    /* MAYBE SEPARATE GPU AND CPU UNIT TESTS??? DAMN BRUTHA*/

    // GPU Convolution test
    {
        //GPU::to_grayscale<<<
    }

    free(h_img_gray);
    free(h_img_conv);
}

void test_diff_CPU(u_char* h_img_1, u_char* h_img_2, int width, int height, int n_channels)
{
    u_char* img_gray_1 = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    u_char* img_gray_2 = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    u_char* img_dst   = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));

    CPU::to_grayscale(h_img_1, img_gray_1, width, height, n_channels);
    CPU::to_grayscale(h_img_2, img_gray_2, width, height, n_channels);

    CPU::compute_difference(img_gray_1, img_gray_2, img_dst, width, height);
    stbi_write_jpg("../out_diff_CPU.jpeg", width, height, 1, img_dst, width);
    spdlog::info("[CPU] Successfully computed image difference.");

    free(img_gray_1);
    free(img_gray_2);
    free(img_dst);
}

void test_diff_GPU(u_char* d_img_1, u_char* d_img_2, int width, int height, size_t pitch, int n_channels)
{

}