#include "test.hpp"

#include <spdlog/spdlog.h>

#include "img_io.hpp"
#include "img_operations.hpp"

void test_diff_CPU(unsigned char* h_img_1, unsigned char* h_img_2, int width, int height, int n_channels)
{
    auto h_img_1_cpy = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    auto h_img_2_cpy = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));

    // Go to grayscale as diff is meant to be used in grayscale
    CPU::to_grayscale(h_img_1, h_img_1_cpy, width, height, n_channels);
    CPU::to_grayscale(h_img_2, h_img_2_cpy, width, height, n_channels);

    auto h_img_diff = static_cast<u_char*>(calloc(width * height, sizeof(u_char)));
    CPU::compute_difference(h_img_1_cpy, h_img_2_cpy, h_img_diff, width, height);
    write_image("../out_diff_CPU.jpeg", width, height, 1, h_img_diff);

    spdlog::info("[CPU] Successfully computed images difference.");

    free(h_img_diff);
    free(h_img_1_cpy);
    free(h_img_2_cpy);
}

void test_diff_GPU(unsigned char* d_img_1, unsigned char* d_img_2, int width, int height, int pitch, int n_channels)
{}
