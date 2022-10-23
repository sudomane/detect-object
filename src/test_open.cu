#include "test.hpp"

#include <spdlog/spdlog.h>

#include "img_io.hpp"

void test_open_CPU(unsigned char* h_img_1, unsigned char* h_img_2, int width, int height)
{
    write_image("../out_open_CPU_1.jpeg", width, height, 3, h_img_1);
    write_image("../out_open_CPU_2.jpeg", width, height, 3, h_img_2);

    spdlog::info("[CPU] Successfully saved opened images.");
}
