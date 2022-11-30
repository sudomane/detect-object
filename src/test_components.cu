#include "test.hpp"

#include <spdlog/spdlog.h>

#include "error.hpp"
#include "img_io.hpp"
#include "img_operations.hpp"

#define IMG_MORPH_CPU "./CPU_out_morph_opening_closing.jpeg"
#define IMG_THRESHOLD_CPU "./CPU_out_threshold.jpeg"

#define  IMG_MORPH_GPU "" // TODO

void test_threshold_CPU(int threshold)
{
    int width, height;

    auto img_threshold = load_image(IMG_MORPH_CPU, &width, &height, nullptr, true);

    if (img_threshold == nullptr)
        abortError("File not found. Aborting test.");

    // CPU diff test
    {
        CPU::threshold(img_threshold, width, height, threshold);
        write_image(IMG_THRESHOLD_CPU, width, height, 1, img_threshold);

        spdlog::info("[CPU] Successfully computed image threshold.");
    }

    free(img_threshold);
}

void test_connected_components_CPU()
{
    int width, height;

    auto img_components = load_image(IMG_THRESHOLD_CPU, &width, &height, nullptr, true);

    if (img_components == nullptr)
        abortError("File not found. Aborting test.");

    // CPU diff test
    {
        CPU::connected_components(img_components, width, height);
        write_image("./CPU_out_components.jpeg", width, height, 1, img_components);

        spdlog::info("[CPU] Successfully computed connected components.");
    }

    free(img_components);
}