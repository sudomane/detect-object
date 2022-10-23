#ifndef TEST_HPP
#define TEST_HPP

/* Unit test functions header */
void test_open_CPU(unsigned char* h_img_1, unsigned char* h_img_2, int width, int height);

void test_grayscale_CPU(unsigned char* h_img, int width, int height, int n_channels);
void test_grayscale_GPU(unsigned char* d_img, int width, int height, int n_channels, int pitch);

void test_conv_2D_CPU(unsigned char* h_img, int width, int height, int n_channels);
void test_conv_2D_GPU(unsigned char* d_img, int width, int height, int n_channels, int pitch);

void test_diff_CPU(unsigned char* h_img_1, unsigned char* h_img_2, int width, int height, int n_channels);
void test_diff_GPU(unsigned char* d_img_1, unsigned char* d_img_2, int width, int height, int n_channels, int pitch);

#endif