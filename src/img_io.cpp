#include "img_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

unsigned char* load_image(const char* filename, int* x, int* y, int* n_chan, bool load_gray)
{
    if (load_gray)
        return stbi_load(filename, x, y, n_chan, 1);
    return stbi_load(filename, x, y, n_chan, 0);
}

void write_image(const char* filename, int x, int y, int channels, unsigned char* data)
{
    int rc = stbi_write_jpg(filename, x, y, channels, data, x);
}
