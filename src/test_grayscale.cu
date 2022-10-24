#include "test.hpp"

#include <spdlog/spdlog.h>

#include "error.hpp"
#include "img_io.hpp"
#include "img_operations.hpp"

void test_grayscale_CPU(const char* input, const char* output)
{
    int width, height, n_chan;
    u_char* img = load_image(input, &width, &height, &n_chan, false);
    
    if (img == nullptr)
        abortError("File not found. Aborting test.");
    
    u_char* img_gray = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    
    // CPU grayscale test
    {
        CPU::to_grayscale(img, img_gray, width, height, n_chan);
        write_image(output, width, height, 1, img_gray);
    
        spdlog::info("[CPU] Successfully converted image to grayscale.");
    }

    free(img);
    free(img_gray);
}


void test_grayscale_GPU(const char* input, const char* output)
{
    int width, height, n_chan;
    
    auto h_img      = load_image(input, &width, &height, &n_chan, false);    
    auto h_img_gray = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    
    size_t host_pitch_RGB  = width * n_chan * sizeof(u_char);
    size_t host_pitch_gray = width * sizeof(u_char);
 
    cudaError_t rc = cudaSuccess;
    
    size_t  dev_pitch_RGB;
    size_t  dev_pitch_gray;
    u_char* d_img;
    u_char* d_img_gray;

    // Device allocation
    {
        // Allocate device buffers

        rc = cudaMallocPitch(&d_img, &dev_pitch_RGB, width * n_chan * sizeof(u_char), height);
        if (rc)
            abortError("Failed device image RGB allocation.");
        
        rc = cudaMallocPitch(&d_img_gray, &dev_pitch_gray, width * sizeof(u_char), height);
        if (rc)
            abortError("Failed device image gray allocation.");

        // Copy images to buffer

        rc = cudaMemcpy2D(d_img, dev_pitch_RGB, h_img, host_pitch_RGB,
                          width * n_chan * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
            abortError("Failed to copy image RGB to device.");
    }

    // GPU Grayscale test
    {
        int block_size  = 32;
        int w           = std::ceil((float)width / block_size);
        int h           = std::ceil((float)height / block_size);
    
        dim3 dimGrid(w, h);
        dim3 dimBlock(block_size, block_size);
        
        GPU::to_grayscale<<<dimGrid, dimBlock>>>(d_img, d_img_gray, width, height,
                                                 dev_pitch_RGB, dev_pitch_gray, n_chan);
        rc = cudaDeviceSynchronize();
        if (rc)
            abortError("Kernel failed.");

        rc = cudaMemcpy2D(h_img_gray, host_pitch_gray, d_img_gray, dev_pitch_gray,
                          width * sizeof(u_char), height, cudaMemcpyDeviceToHost);
        if (rc)
            abortError("Failed to copy image gray from device to host.");

        write_image(output, width, height, 1, h_img_gray);
        spdlog::info("[GPU] Successfully converted image to grayscale.");
    }

    free(h_img);
    free(h_img_gray);
    cudaFree(d_img);
    cudaFree(d_img_gray);
}
