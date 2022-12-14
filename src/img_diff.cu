#include "img_operations.hpp"

namespace CPU
{
void compute_difference(const u_char* img_1, const u_char* img_2, u_char* dst, int width, int height)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            u_char g0 = img_1[j*width+i];
            u_char g1 = img_2[j*width+i];

            u_char val = abs(g0 - g1);
            
            dst[j * width + i] = val;
        }
    }
}
} // namespace CPU

namespace GPU
{
__global__ void compute_difference(const u_char* img_1, const u_char* img_2, u_char* dst, int width, int height, int pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    u_char g0 = img_1[y * pitch + x];
    u_char g1 = img_2[y * pitch + x];

    u_char val = abs(g0 - g1);

    dst[y * pitch + x] = val;
}
} // namespace GPU