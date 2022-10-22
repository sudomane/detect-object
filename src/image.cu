#include "image.hpp"

#include <stdio.h>
#include <stdlib.h>

/* CPU API */

namespace CPU
{
void to_grayscale(const u_char* src, u_char* dst, int width, int height, int n_channels)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            u_char R = src[(i * height + j) * n_channels];
            u_char G = src[(i * height + j) * n_channels + 1];
            u_char B = src[(i * height + j) * n_channels + 2];
            
            dst[i * height + j] = (R + G + B) / 3.f;
        }
    }
}

void conv_2D(const u_char* src, u_char* dst, int width, int height)
{
    u_char filter[9] = { 1, 2, 1,
                         2, 4, 2,
                         1, 2, 1};

    u_char top_left, top, top_right; // x x x
    u_char mid_left, mid, mid_right; // x o x
    u_char bot_left, bot, bot_right; // x x x

    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {
            top_left   = src[(i-1) * height + j - 1]   * filter[0];
            top        = src[(i-1) * height + j]       * filter[1];
            top_right  = src[(i-1) * height + j + 1]   * filter[2];

            mid_left   = src[i * height + j - 1]       * filter[3];
            mid        = src[i * height + j]           * filter[4];
            mid_right  = src[i * height + j + 1]       * filter[5];

            bot_left   = src[(i+1) * height + 3 - 1]   * filter[6];
            bot        = src[(i+1) * height + j]       * filter[7];
            bot_right  = src[(i+1) * height + j + 1]   * filter[8];

            dst[i * height + j] = top_left + top + top_right + 
                                  mid_left + mid + mid_right +
                                  bot_left + bot + bot_right;
        }
    }
}
}; // namespace CPU

/* GPU API */

namespace GPU
{
__global__ void to_grayscale(u_char* src, u_char* dst, int width, int height, int spitch, int dpitch, int n_channels)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    u_char* src_ptr = src + y * spitch;
    u_char* dst_ptr = dst + y * dpitch;

    dst_ptr[x] = (src_ptr[x * n_channels]   // R
                + src_ptr[x * n_channels + 1]   // G
                + src_ptr[x * n_channels + 2])  // B
                / 3.f; 
}
}; // namespace GPU