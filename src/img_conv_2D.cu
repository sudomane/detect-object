#include <vector>
#include "img_operations.hpp"


// Calcul static du gaussien 2D en un point
static double gaussian(double x, double mu, double sigma ) {
    const double a = ( x - mu ) / sigma;
    return std::exp( -0.5 * a * a );
}

// Génération du kernel de la taille souhaitée
static std::vector<double> gen_kernel(int kernel_size, double sigma) {
    auto kernel2d = std::vector<double>(kernel_size * kernel_size, 0);
    double sum, x;
    // compute values
    for (int row = 0; row < kernel_size; row++)
        for (int col = 0; col < kernel_size; col++) {
            x = gaussian(row, kernel_size / 2., sigma)
                       * gaussian(col, kernel_size / 2., sigma);
            kernel2d[row * kernel_size + col] = x;
            sum += x;
        }
    // normalize
    for (int row = 0; row < kernel_size; row++)
        for (int col = 0; col < kernel_size; col++)
            kernel2d[row * kernel_size + col] /= sum;
    return kernel2d;
}


namespace CPU
{
void conv_2D(const u_char* src, u_char* dst, int width, int height, int kernel_size, double sigma)
{

    int ii, jj;
    float sum;
    std::vector<double> kernel = gen_kernel(kernel_size, sigma);
    int center = (kernel_size - 1) / 2;
    for (int i = center; i < height - center; i++)
    {
        for (int j = center; j < width - center; j++)
        {
            //Convolution moche en 2D directement (c'est long):
            sum = 0;
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    ii = i + ki - center;
                    jj = j + kj - center;
                    sum += src[ii * width + jj] * kernel[ki * kernel_size + kj];
                }
            }
            dst[i * width + j] = sum;
        }
    }
}
} // namespace CPU


namespace GPU
{

    // Kernel 3*3 fixed size
__global__ void conv_2D(const u_char* src, u_char* dst, int width, int height, size_t pitch)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width - 1|| y >= height - 1 || x < 1 || y < 1) return;

    // 3x3 Gaussian blur
    float filter[9] = { 16, 8, 16,
                        8,  4, 8,
                        16, 8, 16};

    float top_left, top, top_right; // x x x
    float mid_left, mid, mid_right; // x o x
    float bot_left, bot, bot_right; // x x x

    top_left   = src[(y-1) * pitch + (x-1)]   / filter[0];
    top        = src[(y-1) * pitch + x]       / filter[1];
    top_right  = src[(y-1) * pitch + (x+1)]   / filter[2];

    mid_left   = src[y     * pitch + (x-1)]   / filter[3];
    mid        = src[y     * pitch + x]       / filter[4];
    mid_right  = src[y     * pitch + (x+1)]   / filter[5];

    bot_left   = src[(y+1) * pitch + (x-1)]   / filter[6];
    bot        = src[(y+1) * pitch + x]       / filter[7];
    bot_right  = src[(y+1) * pitch + (x+1)]   / filter[8];

    dst[y * pitch + x] =(u_char) (top_left + top + top_right + 
                                  mid_left + mid + mid_right +
                                  bot_left + bot + bot_right);    

}

// Dynamic size kernel
__global__ void conv_2D_2(const u_char* src, u_char* dst, int width, int height, size_t pitch, const double* kernel, int kernel_size) {
    int id = blockDim.x * blockIdx.x + threadIdx.x; // id = i * width + j, 1 thread per pixel
    int center = (kernel_size - 1) / 2;

    // Vérification des cas aux bords de l'image : ignorer
    if ((id % width < center) ||
            (id % width >= width - center) ||
            (id / width < center) ||
            (id / width >= height - center)) {
        return;
    }
    int i = id / width;
    int j = id % width;
    int ii, jj;
    // Calcul de la convolution
    for (int ki = 0; ki < kernel_size; ki++) {
        for (int kj = 0; kj < kernel_size; kj++) {
            ii = i + ki - center;
            jj = j + kj - center;
            dst[id] += src[ii * width + jj] * kernel[ki * kernel_size + kj];
        }
    }
}
} // namespace GPU