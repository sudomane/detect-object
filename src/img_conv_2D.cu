#include "img_operations.hpp"

namespace CPU
{
void conv_2D(const u_char* src, u_char* dst, int width, int height)
{
    // 3x3 Gaussian blur
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

    // 5x5 Gaussian blur
    /*
    float ttop_ll, ttop_l, ttop, ttop_r, ttop_rr; // x x x x x
    float top_ll,  top_l,  top,  top_r,  top_rr;  // x x x x x
    float mid_ll,  mid_l,  mid,  mid_r,  mid_rr;  // x x o x x
    float bot_ll,  bot_l,  bot,  bot_r,  bot_rr;  // x x x x x
    float bbot_ll, bbot_l, bbot, bbot_r, bbot_rr; // x x x x x

    for (int i = 2; i < width-2; i++)
    {
        for (int j = 2; j < height-2; j++)
        {
            ttop_ll = src[(j - 2) * width + i - 2]          / 273;
            ttop_l  = (src[(j - 2) * width + i - 1] * 4)    / 273;
            ttop    = (src[(j - 2) * width + i] * 7)        / 273;
            ttop_r  = (src[(j - 2) * width + i + 1] * 4)    / 273;
            ttop_rr = src[(j - 2) * width + i + 2]          / 273;

            top_ll  = (src[(j - 1) * width + i - 2] * 4)    / 273;
            top_l   = (src[(j - 1) * width + i - 1] * 16)   / 273;
            top     = (src[(j - 1) * width + i] * 26)       / 273;
            top_r   = (src[(j - 1) * width + i + 1] * 16)   / 273;
            top_rr  = (src[(j - 1) * width + i + 2] * 4)    / 273;

            mid_ll  = (src[j * width + i - 2] * 7 )         / 273;
            mid_l   = (src[j * width + i - 1] * 26)         / 273;
            mid     = (src[j * width + i] * 41)             / 273;
            mid_r   = (src[j * width + i + 1] * 26)         / 273;
            mid_rr  = (src[j * width + i + 2] * 7 )         / 273;

            bot_ll  = (src[(j + 1) * width + i - 2] * 4)    / 273;
            bot_l   = (src[(j + 1) * width + i - 1] * 16)   / 273;
            bot     = (src[(j + 1) * width + i] * 26)       / 273;
            bot_r   = (src[(j + 1) * width + i + 1] * 16)   / 273;
            bot_rr  = (src[(j + 1) * width + i + 2] * 4)    / 273;

            bbot_ll = src[(j + 2) * width + i - 2]          / 273;
            bbot_l  = (src[(j + 2) * width + i - 1] * 4)    / 273;
            bbot    = (src[(j + 2) * width + i] * 7)        / 273;
            bbot_r  = (src[(j + 2) * width + i + 1] * 4)    / 273;
            bbot_rr = src[(j + 2) * width + i + 2]          / 273;

            u_char sum = (u_char) ttop_ll + ttop_l + ttop + ttop_r + ttop_rr +
                                  top_ll  + top_l  + top  + top_r  + top_rr  +
                                  mid_ll  + mid_l  + mid  + mid_r  + mid_rr  +
                                  bot_ll  + bot_l  + bot  + bot_r  + bot_rr  +
                                  bbot_ll + bbot_l + bbot + bbot_r + bbot_rr;

            dst[j * width + i] = sum;
        }
    }
    */
}
} // namespace CPU


namespace GPU
{
__global__ void conv_2D(const u_char* src, u_char* dst, int width, int height, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width - 1|| y >= height - 1 || x < 1 || y < 1) return;

    // 3x3 Gaussian blur
    float filter[9] = { 16, 8, 16,
                        8,  4, 8,
                        16, 8, 16};

    float top_left, top, top_right; // x x x
    float mid_left, mid, mid_right; // x o x
    float bot_left, bot, bot_right; // x x x

    top_left   = src[(y-1) * pitch + (x-1)]   / filter[0];
    top        = src[(y-1) * pitch + x]       / filter[1];
    top_right  = src[(y-1) * pitch + (x+1)]   / filter[2];

    mid_left   = src[y     * pitch + (x-1)]   / filter[3];
    mid        = src[y     * pitch + x]       / filter[4];
    mid_right  = src[y     * pitch + (x+1)]   / filter[5];

    bot_left   = src[(y+1) * pitch + (x-1)]   / filter[6];
    bot        = src[(y+1) * pitch + x]       / filter[7];
    bot_right  = src[(y+1) * pitch + (x+1)]   / filter[8];

    dst[y * pitch + x] =(u_char) (top_left + top + top_right + 
                                  mid_left + mid + mid_right +
                                  bot_left + bot + bot_right);    

}
} // namespace GPU