#ifndef IMAGE_HPP
#define IMAGE_HPP

typedef unsigned char u_char;

// CPU API

u_char* to_grayscale_CPU(const u_char* img_in, int width, int height);
void gaussian_filter_CPU(u_char* img);

// GPU API
// FIXME

#endif