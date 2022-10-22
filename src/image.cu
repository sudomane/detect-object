#include "image.hpp"

#include <stdio.h>
#include <stdlib.h>

/* CPU API */

namespace CPU
{
void to_grayscale(const u_char* src, u_char* dst, int width, int height, int n_channels)
{
    if (n_channels < 3) return;
    
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            u_char R = src[(j * width + i) * n_channels];
            u_char G = src[(j * width + i) * n_channels + 1];
            u_char B = src[(j * width + i) * n_channels + 2];
            
            dst[j * width + i] = (R + G + B) / 3.f;
        }
    }
}

void conv_2D(const u_char* src, u_char* dst, int width, int height)
{
    float filter[9] = { 16, 8, 16,
                        8,  4, 8,
                        16, 8, 16};

    float top_left, top, top_right; // x x x
    float mid_left, mid, mid_right; // x o x
    float bot_left, bot, bot_right; // x x x

    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {
            top_left   = src[(j-1) * width + (i-1)]   / filter[0];
            top        = src[(j-1) * width + i]       / filter[1];
            top_right  = src[(j-1) * width + (i+1)]   / filter[2];

            mid_left   = src[j     * width + (i-1)]   / filter[3];
            mid        = src[j     * width + i]       / filter[4];
            mid_right  = src[j     * width + (i+1)]   / filter[5];

            bot_left   = src[(j+1) * width + (i-1)]   / filter[6];
            bot        = src[(j+1) * width + i]       / filter[7];
            bot_right  = src[(j+1) * width + (i+1)]   / filter[8];

            dst[j * width + i] =(u_char) (top_left + top + top_right + 
                                          mid_left + mid + mid_right +
                                          bot_left + bot + bot_right);
        }
    }
}
}; // namespace CPU

/* GPU API */

namespace GPU
{
__global__ void to_grayscale(const u_char* src, u_char* dst, int width, int height,
                             size_t spitch, size_t dpitch, int n_channels)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height || n_channels < 3)
        return;

    const u_char* src_ptr = src + y * spitch;
    u_char*       dst_ptr = dst + y * dpitch;

    dst_ptr[x] = (src_ptr[x * n_channels]       // R
                + src_ptr[x * n_channels + 1]   // G
                + src_ptr[x * n_channels + 2])  // B
                / 3.f; 
}

__global__ void conv_2D(const u_char* src, u_char* dst, int width, int height, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;


}
}; // namespace GPU