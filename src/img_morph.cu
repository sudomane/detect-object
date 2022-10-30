#include <array>
#include <vector>
#include <algorithm>
#include "img_operations.hpp"


namespace CPU
{

// 2 Consecutives 1D kernel application with temporary memory allocation to store values
void morph(const unsigned char* src, unsigned char* dst, int width, int height, bool minimum, int kernel_size)
{
    // Dynamic 1D kernel size
    std::vector<int> buffer(kernel_size);

    // result of first 1D kernel
    auto tmp = static_cast<unsigned char *>(calloc(width * height, sizeof(unsigned char)));

    // Edges initialization : we keep the edges
    memcpy(tmp, src, sizeof(unsigned char) * width * height);
    memcpy(dst, src, sizeof(unsigned char) * width * height);

    // Fix column and compute line kernel from src to tmp [NOT in place]
    // start at kernel_size / 2, so we can fill the complete buffer with the edges
    for (int i = 0; i < width; i++)
    {
        // Buffer initialization. WARNING No check on kernel size > width or height
        for (int i_edges = 0; i_edges < kernel_size; i_edges++)
        {
            buffer[i_edges] = src[i_edges * width + i];
        }
        // Process a column :
        for (int j = kernel_size / 2; j < height - kernel_size / 2; j++)
        {
            // Compute the value from the buffer
            if (minimum)
            {
                tmp[j * width + i] = *std::min_element(buffer.begin(), buffer.end());
            }
            else
            {
                tmp[j * width + i] = *std::max_element(buffer.begin(), buffer.end());
            }
            // Add the next value : the one at position + kernel_size / 2 overwritten on the previous one
            // (same position % kernel_size)
            buffer[(j + kernel_size / 2 + 1) % kernel_size] = src[(j + kernel_size / 2 + 1) * width + i];
        }
        // Add the last value
        if (minimum)
        {
            tmp[(height - 1 - kernel_size / 2) * width + i] = *std::min_element(buffer.begin(), buffer.end());
        }
        else
        {
            tmp[(height - 1 - kernel_size / 2) * width + i] = *std::max_element(buffer.begin(), buffer.end());
        }
    }
    // 2nd 1D Kernel
    // Compute per line kernel from tmp to dst
    // Process a column
    for (int j = 0; j < height; j++)
    {
        // Buffer initialization. WARNING No check on kernel size > width or height
        for (int i_edges = 0; i_edges < kernel_size; i_edges++)
        {
            buffer[i_edges] = tmp[j * width + i_edges];
        }
        for (int i = kernel_size / 2; i < width - kernel_size / 2; i++)
        {
            // Compute the value from the buffer
            if (minimum)
            {
                dst[j * width + i] = *std::min_element(buffer.begin(), buffer.end());
            } else
            {
                dst[j * width + i] = *std::max_element(buffer.begin(), buffer.end());
            }

            // Add the next value
            // [ 0 1 2 ] 3 -> 0 [ 1 2 3 ]
            buffer[(i + kernel_size / 2 + 1) % kernel_size] = tmp[j * width + i + kernel_size / 2 + 1];
        }
        // Add the last value
        if (minimum)
        {
            dst[j * width + width - kernel_size / 2 - 1] = *std::min_element(buffer.begin(), buffer.end());
        }
        else
        {
            dst[j * width + width - kernel_size / 2 - 1] = *std::max_element(buffer.begin(), buffer.end());
        }
    }
    free(tmp);
}

}