//#include "object_detection.hpp"
/*
static __device__ void
img_diff(u_char* img1, u_char* img2, u_char* dst, unsigned int pitch, int height, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    u_char g0 = img1[y * pitch + x];
    u_char g1 = img2[y * pitch + x];
    
    u_char val = abs(g0 - g1);
    
    dst[y * pitch + x] = val;
}

static __device__ void
connected_components_kernel(unsigned char *d_img, int *d_label, unsigned int pitch, int width, int height, int global_label) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (d_img[y * pitch + x] == 0) {
        d_label[y * pitch + x] = 0;
        return;
    }

    int top = d_label[(y - 1) * pitch + x];
    int left = d_label[y * pitch + x - 1];

    if (top == 0 && left == 0) {
        d_label[y * pitch + x] = atomicAdd(&global_label, 1);
    } else if (top == 0) {
        d_label[y * pitch + x] = left;
    } else if (left == 0) {
        d_label[y * pitch + x] = top;
    } else if (top != left) {
        d_label[y * pitch + x] = min(top, left);
        int max_label = max(top, left);
        atomicMin(&d_label[(max_label - 1) * pitch], min(top, left));
    } else {
        d_label[y * pitch + x] = top;
    }
}

__global__ void object_detection(u_char* img1, u_char* img2, u_char* diff_buffer, int* d_label, int* coords, unsigned int pitch, int height, int width)
{
    int global_label = 1;
    
    img_diff(img1, img2, diff_buffer, pitch, height, width);
    __syncthreads();    
    
    connected_components_kernel(diff_buffer, d_label, pitch, width, height, global_label);
    __syncthreads();

    
}*/