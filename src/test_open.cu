#include "test.hpp"

#include <spdlog/spdlog.h>

#include "img_io.hpp"

void test_open_CPU(unsigned char* h_img_1, unsigned char* h_img_2, int width, int height)
{
    write_image("CPU_out_open_1.jpeg", width, height, 3, h_img_1);
    write_image("CPU_out_open_2.jpeg", width, height, 3, h_img_2);

    spdlog::info("[CPU] Successfully saved opened images.");
}

void test_open_GPU(u_char* d_img_1, u_char* d_img_2, int width, int height, int pitch)
{
    u_char* h_img_1 = static_cast<u_char*>(calloc(width * height, sizeof(u_char) * 3));
    u_char* h_img_2 = static_cast<u_char*>(calloc(width * height, sizeof(u_char) * 3));

    cudaMemcpy2D(h_img_1, width * 3, d_img_1, pitch, width * 3 * sizeof(u_char), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_img_2, width * 3, d_img_2, pitch, width * 3 * sizeof(u_char), height, cudaMemcpyDeviceToHost);

    write_image("GPU_out_open_1.jpeg", width, height, 3, h_img_1);
    write_image("GPU_out_open_2.jpeg", width, height, 3, h_img_2);

    spdlog::info("[GPU] Successfully saved opened images.");

    free(h_img_1);
    free(h_img_2);
}