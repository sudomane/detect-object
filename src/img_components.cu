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
        std::vector<std::set<u_char>> label_matching {};
        // Add "padding" sets for ease of manipulation
        label_matching.emplace_back();
        label_matching.emplace_back();
        std::array<u_char, 4> neighbors{};
        for (auto line = 0; line < height; line++) {
            for (auto column = 0; column < width; column++) {
                index = line * width + column;
                if (buffer[index] == 0){
                    continue; // Background
                }
                std::fill(neighbors.begin(), neighbors.end(), UCHAR_MAX);
                // Check 4 neighbours around + already processed
                if (line > 0 && buffer[index - width]){
                    neighbors[0] = buffer[index - width]; // North pixel
                    if (column > 0 && buffer[index - width - 1]) {
                        neighbors[1] = buffer[index - width - 1]; // North-west pixel
                    }
                    if (column < width - 1 && buffer[index - width + 1]) {
                        neighbors[2] = buffer[index - width + 1]; // North-east pixel
                    }
                }
                if (column > 0 && buffer[index - 1]) {
                    neighbors[3] = buffer[index - 1]; // West pixel
                }
                // Compute minimum neighbour label
                u_char min_label = *std::min_element(neighbors.begin(), neighbors.end());
                // No labelled neighbour
                if (min_label == UCHAR_MAX) {
                    buffer[index] = current_label;
                    current_label++;
                    // Add new label matching
                    // Add new label value to the matching set
                    label_matching.emplace_back();
                    label_matching.back().insert({current_label});
                } else {
                    // Update correspondance
                    for (const auto& value : neighbors) {
                        if (value != UCHAR_MAX) {
                            // Update set with the lowest label
                            label_matching[value].insert(min_label);
                        }
                    }
                    // Labellise the pixel
                    buffer[index] = min_label;
                }
            }
        }

        // SECOND PASS
        for (auto line = 0; line < height; line++) {
            for (auto column = 0; column < width; column++) {
                index = line * width + column;
                if (buffer[index] != 0) {
                    // Not robust but it works. Need to handle nested matching, ex : Label 3 should be labelled 2 but label 2 should be labelled 1
                    buffer[index] = (*label_matching[*label_matching[buffer[index]].upper_bound(0)].upper_bound(0)) * 16; // ADD VALUE FOR DEBUGGING
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