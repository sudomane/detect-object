#ifndef IMAGE_HPP
#define IMAGE_HPP

// TODO: Add function comments

typedef unsigned char u_char;

/* CPU API */
namespace CPU
{
void to_grayscale(const u_char* src, u_char* dst, int width, int height, int n_channels);
void conv_2D(const u_char* src, u_char* dst, int width, int height);
}; // namespace CPU

/* GPU API */ 
namespace GPU
{
__global__ void to_grayscale(u_char* src, u_char* dst,
                             int width, int height,
                             int spitch, int dpitch, int n_channels);
__global__ void conv_2D(const u_char* src, u_char* dst, int width, int height);
}; // namespace GPU
#endif