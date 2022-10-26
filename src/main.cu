#include <stdio.h>
#include <spdlog/spdlog.h>

#include "error.hpp"
#include "test.hpp"
#include "img_io.hpp"
#include "img_operations.hpp"

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        printf("Usage: ./main [image 1] [image 2]\n");
        return -1;
    }
    
    const char* image_1 = argv[1];
    const char* image_2 = argv[2];
    
    u_char* h_img_1;
    u_char* h_img_2;

    int width, height, n_channels;
    int width_, height_, n_channels_;

    // Allocate image on host, for CPU specific tasks
    {
        h_img_1 = load_image(image_1, &width, &height, &n_channels, false);
        if (h_img_1 == nullptr)
            abortError("Could not find image 1");

        h_img_2 = load_image(image_2, &width_, &height_, &n_channels_, false);
        if (h_img_2 == nullptr)
            abortError("Could not find image 2");
            
        if (width != width_ || height != height_ || n_channels != n_channels_)
        {
            spdlog::error("Image dimensions mismatch! {}x{}x{} against {}x{}x{}",
                          width, height, n_channels, width_, height_, n_channels_);
            return -1;
        }
        
        spdlog::info("[CPU] Loaded images {} {} | {}x{}x{}",
                     image_1, image_2, width, height, n_channels);
    }
    
    cudaError_t rc = cudaSuccess;

    u_char* d_img_1;
    u_char* d_img_2;
    size_t pitch;
    
    // Allocate image on device as well, for GPU specific tasks.
    // Since both images are of exact same dimensions, it's ok to overwrite the pitch.
    {
        rc = cudaMallocPitch(&d_img_1, &pitch,
                             width * n_channels * sizeof(u_char), height);
        if (rc)
            abortError("Failed device image 1 allocation.");

        rc = cudaMemcpy2D(d_img_1, pitch, h_img_1, width * n_channels * sizeof(u_char),
                          width * n_channels * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
            abortError("Failed to copy image 1 to device.");
        
        rc = cudaMallocPitch(&d_img_2, &pitch,
                             width * n_channels * sizeof(u_char), height);
        if (rc)
            abortError("Failed device image 2 allocation.");

        rc = cudaMemcpy2D(d_img_2, pitch, h_img_2, width * n_channels * sizeof(u_char),
                          width * n_channels * sizeof(u_char), height, cudaMemcpyHostToDevice);
        if (rc)
            abortError("Failed to copy image 2 to device.");

        spdlog::info("[GPU] Loaded images {} {} | {}x{}x{}",
                     image_1, image_2, width, height, n_channels);
    }



    // Running the tests

    // CPU Tests
    test_open_CPU(h_img_1, h_img_2, width, height);
    test_grayscale_CPU(image_1, "CPU_out_gray_1.jpeg");
    test_grayscale_CPU(image_2, "CPU_out_gray_2.jpeg");
    test_conv_2D_CPU("CPU_out_gray_1.jpeg", "CPU_out_conv_1.jpeg");
    test_conv_2D_CPU("CPU_out_gray_2.jpeg", "CPU_out_conv_2.jpeg");
    test_diff_CPU();
    test_morph_erosion_CPU();
    test_morph_dilation_CPU();
    test_morph_opening_CPU();
    test_morph_closing_CPU();
    test_morph_opening_closing_CPU();

    // GPU Tests
    test_open_GPU(d_img_1, d_img_2, width, height, pitch);
    test_grayscale_GPU(image_1, "GPU_out_gray_1.jpeg");
    test_grayscale_GPU(image_2, "GPU_out_gray_2.jpeg");
    test_conv_2D_GPU("GPU_out_gray_1.jpeg", "GPU_out_conv_1.jpeg");
    test_conv_2D_GPU("GPU_out_gray_2.jpeg", "GPU_out_conv_2.jpeg");
    test_diff_GPU();

    free(h_img_1);
    free(h_img_2);
    cudaFree(d_img_1);
    cudaFree(d_img_2);
    
    return 0;
}
