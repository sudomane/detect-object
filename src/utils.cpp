#include <cstdlib>
#include <complex>
#include "utils.h"

// Calcul static du gaussien 2D en un point
static double gaussian(double x, double mu, double sigma ) {
    const double a = ( x - mu ) / sigma;
    return std::exp( -0.5 * a * a );
}


// Génération du kernel de la taille souhaitée
float* gen_kernel(int kernel_size, double sigma) {
    float *kernel2d = static_cast<float *>(malloc(kernel_size * kernel_size * sizeof(float)));
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