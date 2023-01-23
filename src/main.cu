#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv4/opencv2/opencv.hpp>

#include "object_detection.hpp"

#define WIDTH  640
#define HEIGHT 480

int main() {
    // Open the video capture
    cv::VideoCapture capture(0);

    // Set the video frame size
    capture.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    // Allocate memory on the GPU
    int *d_coords;
    size_t d_pitch;
    
    unsigned char *d_frame;
    unsigned char *d_ref_frame;
    cudaMallocPitch(&d_frame, &d_pitch, WIDTH * sizeof(unsigned char), HEIGHT);
    cudaMallocPitch(&d_ref_frame, &d_pitch, WIDTH * sizeof(unsigned char), HEIGHT);
    
    cudaMalloc(&d_coords, 4 * sizeof(int));

    // Allocate memory on the Host
    int h_coords[4] = {0, 0, 0, 0};

    // Loop through the video frames
    cv::Mat frame, gray_frame;
    cv::Mat ref_frame;
    
    while (capture.read(frame)) {
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

        if (cv::waitKey(1) == 's')
        {
            ref_frame = gray_frame.clone();
            continue;
        }

        cudaMemcpy2D(d_ref_frame, d_pitch, ref_frame.data, d_pitch, WIDTH * sizeof(unsigned char), HEIGHT, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_frame, d_pitch, frame.data, d_pitch, WIDTH * sizeof(unsigned char), HEIGHT, cudaMemcpyHostToDevice);
        cudaMemcpy(&h_coords, d_coords, 4 * sizeof(int), cudaMemcpyDeviceToHost);

        dim3 block(16, 16);
        dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
        //object_detection<<<grid, block>>>(d_frame, d_ref_frame, d_coords, d_pitch);
        
        cv::Point pt1(h_coords[0], h_coords[1]);
        cv::Point pt2(h_coords[2], h_coords[3]);

        // Draw the objects on the frame
        cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 0, 255), 2);

        // Show the frame
        cv::imshow("Object Detection", gray_frame);

        // Exit if the user presses 'q'
        if (cv::waitKey(1) == 'q') break;
    }

    // Release the video capture and GPU memory
    capture.release();
    cudaFree(d_frame);
    cudaFree(d_coords);

    return 0;
}
