#pragma once

typedef unsigned char u_char;

__global__ void object_detection(u_char* img1, u_char* img2, u_char* diff_buffer, int* d_label, int* coords, unsigned int pitch, int height, int width);
