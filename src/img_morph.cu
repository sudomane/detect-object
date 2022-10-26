#include <array>
#include <algorithm>
#include "img_operations.hpp"


namespace CPU
{

// Opening = min, max, Closing = max, min
void morph(const unsigned char* src, unsigned char* dst, int width, int height, int kernel_size, bool minimum)
{

    std::array<int, 9> values{};
    // Gestion sur les bords ?
    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {
            values[0] = src[(j-1) * width + (i-1)];
            values[1] = src[(j-1) * width + i];
            values[2] = src[(j-1) * width + (i+1)];

            values[3] = src[j     * width + (i-1)];
            values[4] = src[j     * width + i];
            values[5] = src[j     * width + (i+1)];

            values[6] = src[(j+1) * width + (i-1)];
            values[7] = src[(j+1) * width + i];
            values[8] = src[(j+1) * width + (i+1)];

            if (minimum) {
                dst[j * width + i] = *std::min_element(values.begin(), values.end());
            } else {
                dst[j * width + i] = *std::max_element(values.begin(), values.end());
            }
        }
    }
}

}