#include <map>
#include <algorithm>
#include <vector>
#include <set>

#include "img_operations.hpp"

namespace CPU
{
    // Thresholding + Two pass algorithm from wikipedia
    void connected_components(u_char* buffer, int width, int height)
    {
        // FIRST PASS
        int index;
        u_char current_label = 2;
        std::map<u_char, u_char> label_matching{{0,0}, {1, 0}}; // Add "padding" sets for ease of manipulation
        std::array<u_char, 4> neighbors{};
        for (auto line = 0; line < height; line++) {
            for (auto column = 0; column < width; column++) {
                index = line * width + column;
                std::fill(neighbors.begin(), neighbors.end(), UCHAR_MAX);
                if (buffer[index] != 1){
                    continue; // Background
                }
                // Check 4 neighbours around + already processed
                if (line > 0 && buffer[index-line]){
                    neighbors[0] = std::min(neighbors[0], buffer[index - line]); // North pixel
                    if (column > 0 && buffer[index - line - 1]) {
                        neighbors[1] = std::min(neighbors[1], buffer[index - line - 1]); // North-west pixel
                    }
                    if (column < width - 1 && buffer[index - line + 1]) {
                        neighbors[2] = std::min(neighbors[2], buffer[index - line + 1]); // North-east pixel
                    }
                }
                if (column > 0 && buffer[index - 1]) {
                    neighbors[3] = std::min(neighbors[3], buffer[index - 1]); // West pixel
                }
                // No other neighbours
                if (*std::min_element(neighbors.begin(), neighbors.end()) == UCHAR_MAX) {
                    buffer[index] = current_label;
                    current_label++;
                    // Add new label matching
                    // Add new label value to the matching
                    label_matching.insert({current_label, current_label});
                } else {
                    // Update correspondance
                    for (const auto& value : neighbors) {
                        if (value != UCHAR_MAX) {
                            // Update map with the lowest label
                            label_matching.find(value)->second = *std::min_element(neighbors.begin(), neighbors.end());
                        }
                    }
                    // Labellise the pixel
                    buffer[index] = *std::min_element(neighbors.begin(), neighbors.end());
                }
            }
        }

        // SECOND PASS
        for (auto line = 0; line < height; line++) {
            for (auto column = 0; column < width; column++) {
                index = line * width + column;
                if (buffer[index] != 0) {
                    buffer[index] = label_matching.find(buffer[index])->second;
                }
            }
        }
    }

    void threshold(u_char* buffer, int width, int height, int threshold) {
        int index;
        for (auto line = 0; line < height; line++) {
            for (auto column = 0; column < width; column++) {
                index = line * width + column;
                if (buffer[index] > threshold) {
                    buffer[index] = 1;
                } else {
                    buffer[index] = 0;
                }
            }
        }
    }
} // namespace CPU

namespace GPU
{
__global__ void connected_components(u_char* buffer, int width, int height)
{}
} // namespace GPU