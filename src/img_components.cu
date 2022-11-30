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
        // Add "padding" sets for ease of manipulation
        std::map<int, int> label_matching{{0,0}, {1,0}, {2, 2}};
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
                    // Add new label matching and value to the matching map
                    label_matching.insert({current_label, current_label});
                } else {
                    // Update correspondance
                    for (const auto& value : neighbors) {
                        if (value != UCHAR_MAX) {
                            // Update set with the lowest label
                            label_matching.find(value)->second = min_label;
                        }
                    }
                    // Labellise the pixel
                    buffer[index] = min_label;
                }
            }
        }
        // Handle nested label value by going from first to last label
        for (auto& match : label_matching) {
            match.second = label_matching.find(match.second)->second;
        }
        // SECOND PASS
        for (auto line = 0; line < height; line++) {
            for (auto column = 0; column < width; column++) {
                index = line * width + column;
                if (buffer[index] != 0) {
                    // Not robust but it works. Need to handle nested matching, ex : Label 3 should be labelled 2 but label 2 should be labelled 1
                    buffer[index] = label_matching.find(buffer[index])->second * 16; // ADD VALUE FOR DEBUGGING
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