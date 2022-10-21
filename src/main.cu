#include <stdio.h>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include "image.hpp"

__global__ void my_kernel()
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Thread n %d\n", i);
}

int main()
{
    int width, height, channels;

    u_char* img = stbi_load("../image.jpeg", &width, &height, &channels, 0);

    if (img == nullptr) return -1;

    printf("Loaded image %dx%dx%d\n", width, height, channels);

    u_char* img_gray = to_grayscale_CPU(img, width, height);
    if (img_gray == nullptr) return -1;

    u_char* img_conv = conv_2D_CPU(img, width, height);
    if (img_conv == nullptr) return -1;

    stbi_write_jpg("../out_gray.jpeg", width, height, 1, img_gray, width);
    stbi_write_jpg("../out_conv.jpeg", width, height, 1, img_conv, width);

    free(img);
    free(img_gray);
    free(img_conv);

    return 0;
}