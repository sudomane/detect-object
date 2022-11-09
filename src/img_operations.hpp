#ifndef IMG_OPERATIONS_HPP
#define IMG_OPERATIONS_HPP

/* Header file for CPU/GPU image operations */

// TODO: Add function comments

typedef unsigned char u_char;

namespace CPU
{
void to_grayscale(const u_char* src, u_char* dst, int width, int height, int n_channels);
void conv_2D(const u_char* src, u_char* dst, int width, int height, int kernel_size, double sigma);
void morph(const u_char* src, u_char* dst, int width, int height, bool minimum, int kernel_size);
void compute_difference(const u_char* img_1, const u_char* img_2, u_char* dst, int width, int height);
void erosion(const u_char* src, u_char* dst, int width, int height);
void dilation(const u_char* src, u_char* dst, int width, int height);
}; // namespace CPU

namespace GPU
{
__global__ void to_grayscale(const u_char* src, u_char* dst, int width, int height,
                             size_t spitch, size_t dpitch, int n_channels);
__global__ void conv_2D(const u_char* src, u_char* dst, int width, int height,
                        size_t pitch);
__global__ void compute_difference(const u_char* img_1, const u_char* img_2, u_char* dst,
                                   int width, int height, int pitch);
__global__ void erosion(const u_char* src, u_char* dst, int width, int height, int pitch);
__global__ void dilation(const u_char* src, u_char* dst, int width, int height, int pitch);
}; // namespace GPU

#endif // IMG_OPERATIONS_HPP