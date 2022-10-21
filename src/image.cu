#include "image.hpp"

#include <stdio.h>
#include <stdlib.h>

/* CPU API */

namespace CPU
{
void to_grayscale(const u_char* src, u_char* dst, int width, int height)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            u_char R = src[(i * height + j) * 3];
            u_char G = src[(i * height + j) * 3 + 1];
            u_char B = src[(i * height + j) * 3 + 2];
            
            dst[i * height + j] = (R + G + B) / 3;
        }
    }
}

void conv_2D(u_char* src, u_char* dst, int width, int height)
{
    u_char filter[9] = { 1, 2, 1,
                        2, 4, 2,
                        1, 2, 1};
                        

    u_char top_left, top, top_right;
    u_char mid_left, mid, mid_right;
    u_char bot_left, bot, bot_right;

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

            bot_left   = src[(i+1) * height + j - 1]   * filter[6];
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
__global__ void to_grayscale(u_char* src, u_char* dst, int width, int height, int pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > width || y > height)
        return;

    u_char* lineptr = src + (y * pitch) * 3;

    u_char R = lineptr[x];
    u_char G = lineptr[x+1];
    u_char B = lineptr[x+2];

    dst[x * pitch + y] = (R + G + B) / 3;
}
}; // namespace GPU