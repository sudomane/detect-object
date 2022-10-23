#ifndef IMAGE_HPP
#define IMAGE_HPP

/* stb_image wrapper */

// The functions in this header are only for loading/writing images.
// Nothing else. We will barely (if not, never) need to modify this file.

unsigned char* load_image(const char* filename, int* x, int* y, int* n_chan);
void           write_image(const char* filename, int x, int y, unsigned char* data);

#endif // IMAGE_HPP