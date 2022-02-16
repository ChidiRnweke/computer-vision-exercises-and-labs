import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def gaussianKernel(dimensions: int, sigma: float) -> np.array:
    center = (dimensions - 1) / 2
    denom = 1 / (2 * np.pi * (sigma ** 2))
    kernel = np.fromfunction(
        lambda x, y: denom
        * np.e ** -(((x - center) ** 2 + (y - center) ** 2) / (2 * (sigma ** 2))),
        shape=(dimensions, dimensions),
    )
    return kernel / np.sum(kernel)


def main():
    hundreds = np.zeros((100, 100))
    twohundreds = np.ones((100, 100)) * 100
    threehunderds = np.ones((100, 100)) * 200

    concatenatedArray = np.concatenate((hundreds, twohundreds, threehunderds), axis=1)
    imagewithnoise = concatenatedArray + 50 * (np.random.rand(100, 300) - 0.5)
    fig, ax = plt.subplots(3)
    ax[0].imshow(imagewithnoise, cmap="gray")
    ax[0].set_title("original noisy image")
    ax[1].imshow(signal.convolve2d(imagewithnoise, gaussianKernel(5, 3)), cmap="gray")
    ax[1].set_title("after applying gaussian kernel with kernel size 5")
    ax[2].imshow(signal.convolve2d(imagewithnoise, gaussianKernel(15, 3)), cmap="gray")
    ax[2].set_title("after applying gaussian kernel with kernel size 15")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.show()


if __name__ == "__main__":
    main()
