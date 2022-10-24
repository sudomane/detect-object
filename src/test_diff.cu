#include "test.hpp"

#include <spdlog/spdlog.h>

#include "error.hpp"
#include "img_io.hpp"
#include "img_operations.hpp"

#define  IMG_CONV_CPU_1 "CPU_out_conv_1.jpeg"
#define  IMG_CONV_CPU_2 "CPU_out_conv_2.jpeg"

#define  IMG_CONV_GPU_1 "GPU_out_conv_1.jpeg"
#define  IMG_CONV_GPU_2 "GPU_out_conv_2.jpeg"

void test_diff_CPU()
{
    int width, height;
    int width_, height_;
    
    auto img_conv_1 = load_image(IMG_CONV_CPU_1, &width, &height, nullptr, true);
    auto img_conv_2 = load_image(IMG_CONV_CPU_2, &width_, &height_, nullptr, true);

    if (img_conv_1 == nullptr || img_conv_2 == nullptr)
        abortError("File not found. Aborting test.");

    if (width != width_ || height != height_)
        abortError("Image dimension mismatch! Aborting.");

    auto img_diff = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));

    // CPU diff test
    {
        CPU::compute_difference(img_conv_1, img_conv_2, img_diff, width, height);
        write_image("./CPU_out_diff.jpeg", width, height, 1, img_diff);

        spdlog::info("[CPU] Successfully computed images difference.");
    }

    free(img_diff);
    free(img_conv_1);
    free(img_conv_2);
}

void test_diff_GPU()
{
    int width, height;
    int width_, height_;
    
    auto h_img_conv_1 = load_image(IMG_CONV_GPU_1, &width, &height, nullptr, true);
    auto h_img_conv_2 = load_image(IMG_CONV_GPU_2, &width_, &height_, nullptr, true);
    
    if (h_img_conv_1 == nullptr || h_img_conv_2 == nullptr)
        abortError("File not found. Aborting test.");

    if (width != width_ || height != height_)
        abortError("Image dimension mismatch! Aborting.");

    auto h_img_diff = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));

    size_t host_pitch = width * sizeof(u_char);

    cudaError_t rc = cudaSuccess;

    size_t  dev_pitch;
    u_char* d_img_diff;
    u_char* d_img_conv_1;
    u_char* d_img_conv_2;

    // Device allocation
    {
        // Allocate device buffers

        rc = cudaMallocPitch(&d_img_diff, &dev_pitch, width * sizeof(u_char), height);
        if (rc)
            abortError("Failed device image diff allocation.");

        rc = cudaMallocPitch(&d_img_conv_1, &dev_pitch, width * sizeof(u_char), height);
        if (rc)
            abortError("Failed device conv 2D image_1 allocation.");
        
        rc = cudaMallocPitch(&d_img_conv_2, &dev_pitch, width * sizeof(u_char), height);
        if (rc)
            abortError("Failed device conv 2D image_2 allocation.");

        // Copy images to buffer

        rc = cudaMemcpy2D(d_img_conv_1, dev_pitch, h_img_conv_1, host_pitch,
                          width * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
            abortError("Failed to copy image_1 to device.");

        rc = cudaMemcpy2D(d_img_conv_2, dev_pitch, h_img_conv_2, host_pitch,
                          width * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
            abortError("Failed to copy image_2 to device.");
    }
    
    // GPU diff test
    {
        int block_size  = 32;
        int w           = std::ceil((float)width / block_size);
        int h           = std::ceil((float)height / block_size);
    
        dim3 dimGrid(w, h);
        dim3 dimBlock(block_size, block_size);

        GPU::compute_difference<<<dimGrid, dimBlock>>>(d_img_conv_1, d_img_conv_2, d_img_diff,
                                                       width, height, dev_pitch);

        rc = cudaDeviceSynchronize();
        if (rc)
            abortError("Kernel failed.");

        rc = cudaMemcpy2D(h_img_diff, host_pitch, d_img_diff, dev_pitch,
                          width * sizeof(u_char), height, cudaMemcpyDeviceToHost);
        if (rc)
            abortError("Failed to copy image from device to host.");

        write_image("./GPU_out_diff.jpeg", width, height, 1, h_img_diff);
    }

    free(h_img_diff);
    free(h_img_conv_1);
    free(h_img_conv_2);
    cudaFree(d_img_diff);
    cudaFree(d_img_conv_1);
    cudaFree(d_img_conv_2);
}
