#include <cassert>
#include <stdio.h>
#include <spdlog/spdlog.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include "image.hpp"

void test_grayscale(const char* file)
{
    int rc;
    int width, height, channels;
    size_t pitch;

    u_char* img = stbi_load(file, &width, &height, &channels, 0);

    if (img == nullptr)
    {
        spdlog::error("Could not find image %s", file);
        return;
    }

    spdlog::info("Loaded image {} | {}x{}x{}", file, width, height, channels);

    int bsize   = 32;
    int w       = std::ceil((float)width / bsize);
    int h       = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    u_char* h_img_gray = static_cast<u_char*>(malloc(width * height * sizeof(u_char)));
    u_char* d_img_gray;
    
    rc = cudaMallocPitch(&d_img_gray, &pitch, width * sizeof(u_char), height);
    if (rc)
    {
        spdlog::error("Failed GPU image allocation. Error code: {}", rc);
        return;
    }

    CPU::to_grayscale(img, h_img_gray, width, height);
    stbi_write_jpg("../out_gray_CPU.jpeg", width, height, 1, h_img_gray, width);
    spdlog::info("[CPU] Successfully converted image to grayscale.");
    
    GPU::to_grayscale<<<dimGrid, dimBlock>>>(img, d_img_gray, width, height, pitch);
    cudaDeviceSynchronize();
    cudaMemcpy2D(h_img_gray, width, d_img_gray, pitch, width * sizeof(u_char), height, cudaMemcpyDeviceToHost);
    
    stbi_write_jpg("../out_gray_GPU.jpeg", width, height, 1, h_img_gray, width);
    spdlog::info("[GPU] Successfully converted image to grayscale.");
    
    free(img);
    free(h_img_gray);
    cudaFree(d_img_gray);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: ./main [path_to_image]\n");
        return -1;
    }
    
    test_grayscale(argv[1]);
    return 0;
}