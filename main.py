# This is a sample Python script.
import sys  # Argument parsing
from skimage import io  # Image opening
from skimage import morphology  # Object recognition
from skimage.draw import rectangle_perimeter  # Rectangle edge computing
from skimage.filters import gaussian  # Filtering
from skimage.color import rgb2gray  # Grayscale conversion
import numpy as np # Array manipulation
from PIL import Image # Image saving


def draw_output(image, vertices):
    bounding_box = rectangle_perimeter((vertices[0], vertices[2]), (vertices[1], vertices[3]))
    output = np.copy(image)
    output[bounding_box] = 1
    return output


def get_bounding_box(label_map):
    x_max, y_max = np.max(np.where(label_map != 0), 1)
    x_min, y_min = np.min(np.where(label_map != 0), 1)
    return x_min, x_max, y_min, y_max


if __name__ == '__main__':
    #  Argument check
    if len(sys.argv) < 3 :
        print(sys.argv[1])
        print("Missing images arguments")
        quit(-1)

    # Image opening
    baseline = io.imread(sys.argv[1])
    new_image = io.imread(sys.argv[2])

    # Diff, grayscale conversion, filtering
    diff = np.abs(baseline - new_image)
    filtered = gaussian(diff)
    grayscale = rgb2gray(filtered)

    # Object segmentation
    label = morphology.binary_closing(morphology.binary_opening(grayscale))

    # Bounding box computing & drawing
    limits = get_bounding_box(label)
    result = draw_output(new_image, limits)

    # Save & quit
    Image.fromarray(result).save("output.jpg")
    quit(0)

