#include "img_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

unsigned char* load_image(const char* filename, int* x, int* y, int* n_chan)
{
    unsigned char* img = stbi_load(filename, x, y, n_chan, 0);
    return img;
}

void write_image(const char* filename, int x, int y, unsigned char* data)
{
    int rc = stbi_write_jpg(filename, x, y, 1, data, x);
}
