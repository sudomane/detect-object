#ifndef TEST_HPP
#define TEST_HPP

/* Unit test functions header */
void test_open_CPU(unsigned char* h_img_1, unsigned char* h_img_2, int width, int height);
void test_open_GPU(unsigned char* d_img_1, unsigned char* d_img_2, int width, int height, int pitch);

void test_grayscale_CPU(const char* input, const char* output);
void test_grayscale_GPU(const char* input, const char* output);

void test_conv_2D_CPU(const char* input, const char* output);
void test_conv_2D_GPU(const char* input, const char* output);

void test_diff_CPU();
void test_diff_GPU();

#endif