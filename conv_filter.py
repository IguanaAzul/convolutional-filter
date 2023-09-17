import numpy as np
import cv2
from skimage.exposure import rescale_intensity


def convolve(image_array, kernel, padding=0, stride=1):
    (image_height, image_width) = image_array.shape
    (kernel_height, kernel_width) = kernel.shape
    convolution_input = (
        cv2.copyMakeBorder(image_array, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        if padding
        else image_array
    )
    output_shape = (
        int(np.floor((image_height - kernel_height + 2 * padding) / stride) + 1),
        int(np.floor((image_width - kernel_width + 2 * padding) / stride) + 1),
    )
    output = np.zeros(output_shape, dtype="float32")
    for y_input, y_output in zip(np.arange(0, convolution_input.shape[0], step=stride), np.arange(output_shape[1])):
        for x_input, x_output in zip(np.arange(0, convolution_input.shape[1], step=stride), np.arange(output_shape[0])):
            slice_of_image = convolution_input[y_input: y_input + kernel_height, x_input: x_input + kernel_width]
            output[x_output, y_output] = (slice_of_image * kernel).sum()
    output = np.absolute(output)
    output = rescale_intensity(output, in_range=(0, 255))
    output = np.transpose((output * 255).astype("uint8"))

    return output


vertical_sobel_filter = np.array(
    (
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ), dtype="int"
)

horizontal_sobel_filter = np.transpose(vertical_sobel_filter)

teste_edger = np.array(
    (
        [-1, -5, -2, 1, 1],
        [-5, -2, -4, 2, 1],
        [-4, -4, 0, 4, 4],
        [-1, -2, 4, 2, 5],
        [-1, -1, 2, 5, 1]
    ), dtype="int"
)

conv_filter_1 = np.array(
    (
        [0, 1, -1, 0],
        [1, 3, -3, -1],
        [1, 3, -3, -1],
        [0, 1, -1, 0]
    ), dtype="int"
)

image = cv2.imread("images/ramon.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

convolveOutput = convolve(gray, teste_edger, padding=100, stride=5)
cv2.imshow("original", gray)
convolveOutput[convolveOutput < 255] = 0
cv2.imshow("CONVOLUCIONADO", convolveOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()