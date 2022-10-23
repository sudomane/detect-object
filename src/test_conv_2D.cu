#include "test.hpp"

#include <spdlog/spdlog.h>

#include "img_io.hpp"
#include "img_operations.hpp"

void test_conv_2D_CPU(unsigned char* h_img, int width, int height, int n_channels)
{
    auto h_img_gray = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));

    // Go to grayscale as diff is meant to be used in grayscale
    CPU::to_grayscale(h_img, h_img_gray, width, height, n_channels);

    auto h_img_conv = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    CPU::conv_2D(h_img_gray, h_img_conv, width, height);
    write_image("../out_conv_CPU.jpeg", width, height, 1, h_img_conv);

    spdlog::info("[CPU] Successfully applied 2D convolution.");

    free(h_img_gray);
    free(h_img_conv);
}

void test_conv_2D_GPU(unsigned char* d_img, int width, int height, int n_channels, int pitch)
{}
