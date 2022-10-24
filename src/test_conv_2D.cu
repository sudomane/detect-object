#include "test.hpp"

#include <spdlog/spdlog.h>

#include "error.hpp"
#include "img_io.hpp"
#include "img_operations.hpp"

void test_conv_2D_CPU(const char* input, const char* output)
{
    int width, height;
    auto img_gray = load_image(input, &width, &height, nullptr, true);
    
    if (img_gray == nullptr)
        abortError("File not found. Aborting test.");

    auto img_conv = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    
    // CPU conv 2D test
    {
        CPU::conv_2D(img_gray, img_conv, width, height);
        write_image(output, width, height, 1, img_conv);

        spdlog::info("[CPU] Successfully applied 2D convolution.");
    }

    free(img_gray);
    free(img_conv);
}

void test_conv_2D_GPU(const char* input, const char* output)
{
    int width, height;
    auto h_img_gray = load_image(input, &width, &height, nullptr, true);
    
    if (h_img_gray == nullptr)
        abortError("File not found. Aborting test.");
    
    auto h_img_conv = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));

    size_t host_pitch = width * sizeof(u_char);
    
    cudaError_t rc = cudaSuccess;

    size_t  dev_pitch;
    u_char* d_img_conv;
    u_char* d_img_gray;

    // Device allocation
    {
        // Allocate device buffers

        rc = cudaMallocPitch(&d_img_conv, &dev_pitch, width * sizeof(u_char), height);
        if (rc)
            abortError("Failed device conv 2D image allocation.");
        
        rc = cudaMallocPitch(&d_img_gray, &dev_pitch, width * sizeof(u_char), height);
        if (rc)
            abortError("Failed device grayscale image allocation.");

        // Copy images to buffer

        rc = cudaMemcpy2D(d_img_gray, dev_pitch, h_img_gray, host_pitch,
                          width * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
            abortError("Failed to copy grayscale image to device.");
    }

    // GPU conv 2D test
    {
        int block_size  = 32;
        int w           = std::ceil((float)width / block_size);
        int h           = std::ceil((float)height / block_size);
    
        dim3 dimGrid(w, h);
        dim3 dimBlock(block_size, block_size);
        
        GPU::conv_2D<<<dimGrid, dimBlock>>>(d_img_gray, d_img_conv, width, height, dev_pitch);
        rc = cudaDeviceSynchronize();
        if (rc)
            abortError("Kernel failed.");

        rc = cudaMemcpy2D(h_img_conv, host_pitch, d_img_conv, dev_pitch,
                          width * sizeof(u_char), height, cudaMemcpyDeviceToHost);
        if (rc)
            abortError("Failed to copy image from device to host.");

        write_image(output, width, height, 1, h_img_conv);
        spdlog::info("[GPU] Successfully applied 2D convolution.");
    }

    free(h_img_gray);
    free(h_img_conv);
    cudaFree(d_img_gray);
    cudaFree(d_img_conv);
}
