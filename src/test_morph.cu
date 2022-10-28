#include "test.hpp"

#include <spdlog/spdlog.h>

#include "error.hpp"
#include "img_io.hpp"
#include "img_operations.hpp"

#define  IMG_DIFF_CPU "./CPU_out_diff.jpeg"

void test_morph_erosion_CPU()
{
    int width, height;

    auto img_gray = load_image(IMG_DIFF_CPU, &width, &height, nullptr, true);

    if (img_gray == nullptr)
        abortError("File not found. Aborting test.");

    auto img_morph = static_cast<u_char *>(calloc(width * height, sizeof(u_char)));

    // CPU diff test
    {
        CPU::morph(img_gray, img_morph, width, height, true);
        write_image("./CPU_out_morph_erosion.jpeg", width, height, 1, img_morph);

        spdlog::info("[CPU] Successfully computed image morph erosion operation.");
    }

    free(img_morph);
    free(img_gray);
}

void test_morph_dilation_CPU()
{
    int width, height;

    auto img_gray = load_image(IMG_DIFF_CPU, &width, &height, nullptr, true);

    if (img_gray == nullptr)
        abortError("File not found. Aborting test.");

    auto img_morph = static_cast<u_char *>(calloc(width * height, sizeof(u_char)));

    // CPU diff test
    {
        CPU::morph(img_gray, img_morph, width, height, false);
        write_image("./CPU_out_morph_dilation.jpeg", width, height, 1, img_morph);

        spdlog::info("[CPU] Successfully computed image morph dilation operation.");
    }

    free(img_morph);
    free(img_gray);
}

void test_morph_opening_CPU()
{
    int width, height;

    auto img_gray = load_image(IMG_DIFF_CPU, &width, &height, nullptr, true);

    if (img_gray == nullptr)
        abortError("File not found. Aborting test.");

    auto img_morph = static_cast<u_char *>(calloc(width * height, sizeof(u_char)));

    // CPU diff test
    {
        CPU::morph(img_gray, img_morph, width, height, true);
        CPU::morph(img_morph, img_gray, width, height, false);
        write_image("./CPU_out_morph_opening.jpeg", width, height, 1, img_gray);

        spdlog::info("[CPU] Successfully computed image morph opening operation.");
    }

    free(img_morph);
    free(img_gray);
}

void test_morph_closing_CPU()
{
    int width, height;

    auto img_gray = load_image(IMG_DIFF_CPU, &width, &height, nullptr, true);

    if (img_gray == nullptr)
        abortError("File not found. Aborting test.");

    auto img_morph = static_cast<u_char *>(calloc(width * height, sizeof(u_char)));

    // CPU diff test
    {
        CPU::morph(img_gray, img_morph, width, height, false);
        CPU::morph(img_morph, img_gray, width, height, true);
        write_image("./CPU_out_morph_closing.jpeg", width, height, 1, img_gray);

        spdlog::info("[CPU] Successfully computed image morph closing operation.");
    }

    free(img_morph);
    free(img_gray);
}

void test_morph_opening_closing_CPU()
{
    int width, height;

    auto img_gray = load_image(IMG_DIFF_CPU, &width, &height, nullptr, true);

    if (img_gray == nullptr)
        abortError("File not found. Aborting test.");

    auto img_morph = static_cast<u_char *>(calloc(width * height, sizeof(u_char)));

    // CPU diff test
    {
        CPU::morph(img_gray, img_morph, width, height, true);
        CPU::morph(img_morph, img_gray, width, height, false);
        CPU::morph(img_gray, img_morph, width, height, false);
        CPU::morph(img_morph, img_gray, width, height, true);
        write_image("./CPU_out_morph_opening_closing.jpeg", width, height, 1, img_gray);

        spdlog::info("[CPU] Successfully computed image morph opening closing operation.");
    }
    free(img_morph);
    free(img_gray);
}