#include "image.hpp"


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
} // namespace CPU

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
} // namespace GPU