#include "image.hpp"

#include <stdio.h>
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

u_char* conv_2D_CPU(u_char* src, int width, int height)
{
    u_char* dst = static_cast<u_char*>(malloc(width * height * sizeof(u_char)));
    
    if (dst == nullptr) return nullptr;
    
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

    return dst;
}
