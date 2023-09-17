import numpy as np
import cv2
from skimage.exposure import rescale_intensity
import warnings
np.random.seed(8)


def convolve(image_array, kernel, padding=0, stride=1):
    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel.shape
    warned_shape = False
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
    it = 0
    for y_input, y_output in zip(np.arange(0, convolution_input.shape[0], step=stride), np.arange(output_shape[1])):
        it += 1
        for x_input, x_output in zip(np.arange(0, convolution_input.shape[1], step=stride), np.arange(output_shape[0])):
            slice_of_image = convolution_input[y_input: y_input + kernel_height, x_input: x_input + kernel_width]
            if slice_of_image.shape != kernel.shape:
                if not warned_shape:
                    warnings.warn(
                        "This filter isn't processing all pixels of the image "
                        "due to filter shape and image size ratio, "
                        "consider including some padding."
                    )
                    warned_shape = True
                continue
            output[x_output, y_output] = (slice_of_image * kernel).sum()

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

negative_vertical_sobel = -vertical_sobel_filter

horizontal_sobel_filter = np.transpose(vertical_sobel_filter)

negative_horizontal_sobel = -horizontal_sobel_filter

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

crazy_filter = np.random.random((np.random.randint(10, size=2)+1)) * 10 - 5
print(crazy_filter)

image = cv2.imread("images/ramon.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("original", gray)
cv2.imshow("crazy_filter", convolve(gray, crazy_filter))
cv2.imshow("vertical_sobel_filter", convolve(gray, vertical_sobel_filter))
cv2.imshow("horizontal_sobel_filter", convolve(gray, horizontal_sobel_filter))
cv2.imshow("teste_edger", convolve(gray, teste_edger))
cv2.imshow("conv_filter_1", convolve(gray, conv_filter_1))
cv2.imshow("negative_vertical_sobel", convolve(gray, negative_vertical_sobel))
cv2.imshow("negative_horizontal_sobel", convolve(gray, negative_horizontal_sobel))
cv2.imshow("teste_edger padding 100 stride 5", convolve(gray, teste_edger, padding=100, stride=5))
cv2.imshow("teste_edger padding 1 stride 2", convolve(gray, teste_edger, padding=1, stride=2))
cv2.imshow("teste_edger padding 100 stride 5", convolve(gray, teste_edger, padding=100, stride=5))
cv2.waitKey(0)
cv2.destroyAllWindows()
