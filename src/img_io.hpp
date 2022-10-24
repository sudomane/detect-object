#ifndef IMG_IO_HPP
#define IMG_IO_HPP

/* stb_image wrapper */

// The functions in this header are only for loading/writing images.
// Nothing else. We will barely (if not, never) need to modify this file.

unsigned char* load_image(const char* filename, int* x, int* y, int* n_chan, bool load_gray);
void           write_image(const char* filename, int x, int y, int channels, unsigned char* data);

#endif // IMG_IO_HPP