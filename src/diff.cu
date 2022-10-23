#include "img_operations.hpp"

namespace CPU
{
void compute_difference(const u_char* img_1, const u_char* img_2, u_char* dst, int width, int height)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int g0 = static_cast<int>(img_1[j*width+i]);
            int g1 = static_cast<int>(img_2[j*width+i]);
            
            int val = abs(g0 - g1);
            
            dst[j * width + i] = val;
        }
    }
}
} // namespace CPU