#ifndef IMAGE_HPP
#define IMAGE_HPP

typedef unsigned char u_char;

// CPU API

u_char* to_grayscale_CPU(const u_char* img_in, int width, int height);
u_char* conv_2D_CPU(u_char* src, int width, int height);

// GPU API
// FIXME

#endif