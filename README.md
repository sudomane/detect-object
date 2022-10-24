# Object detection with CUDA

The goal of this project is to implement a GPU parallelized object detector from (almost) scratch in CUDA.

## Running the project

```
mkdir build
cd build
cmake ..
make
./main [path to image]
```

## Roadmap

* [x] Load images
* [ ] Apply bounding box on results
* [ ] Display image

### Python prototype
* [x] CPU prototype

### CPU Roadmap
* [x] CPU Grayscale
* [x] CPU 2D Gaussian blur
* [x] CPU Image difference
* [ ] CPU Morphological closing/opening
* [ ] CPU Threshold to keep connected components with high peaks

### GPU Roadmap
* [x] GPU Grayscale
* [x] GPU Convolution
* [x] GPU Image difference
* [ ] GPU Morphological closing/opening
* [ ] GPU Threshold to keep connected components with high peaks
