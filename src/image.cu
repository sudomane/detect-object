#include "image.hpp"

#include <stdlib.h>

/* 
** CPU API
* 
* to_grayscale_CPU
* gaussian_filter_CPU // FIXME
*/

u_char* to_grayscale_CPU(const u_char* img_in, int width, int height)
{
    u_char* img_gray = static_cast<u_char*>(malloc(width * height * sizeof(u_char)));

    if (img_gray == nullptr) return nullptr;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            u_char R = img_in[(i * height + j) * 3];
            u_char G = img_in[(i * height + j) * 3 + 1];
            u_char B = img_in[(i * height + j) * 3 + 2];
            
            img_gray[i * height + j] = (R + G + B) / 3;
        }
    }

    return img_gray;
}

void gaussian_filter_CPU(u_char* img)
{
    u_char filter[9] = { 1, 2, 1,
                         2, 4, 2,
                         1, 2, 1};
}
