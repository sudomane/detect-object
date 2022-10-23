#include <stdio.h>
#include <spdlog/spdlog.h>

#include "test.hpp"
#include "image.hpp"
#include "img_operations.hpp"

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
        h_img_1 = load_image(argv[1], &width, &height, &n_channels);
        if (h_img_1 == nullptr)
        {
            spdlog::error("Could not find image {}", argv[1]);
            return -1;
        }

        h_img_2 = load_image(argv[2], &width_, &height_, &n_channels_);
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
        
        spdlog::info("[CPU] Loaded images {} {} | {}x{}x{}",
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
    
    test_grayscale_CPU(h_img_1, width, height, n_channels);
    test_grayscale_GPU(d_img_1, width, height, n_channels, pitch);
    
    test_conv_2D_CPU(h_img_1, width, height, n_channels);
    test_conv_2D_GPU(d_img_1, width, height, n_channels, pitch);
    
    test_diff_CPU(h_img_1, h_img_2, width, height, n_channels);
    test_diff_GPU(d_img_1, d_img_2, width, height, n_channels, pitch);

    free(h_img_1);
    free(h_img_2);
    cudaFree(d_img_1);
    cudaFree(d_img_2);
    
    return 0;
}
