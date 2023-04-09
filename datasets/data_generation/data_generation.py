import os
import random
from PIL import Image
import numpy as np
import cv2
from scipy.signal import convolve2d


class ReflectionSythesis_1(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of ICCV 2017 paper *"A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing"*    
    """

    def __init__(self, kernel_sizes=None, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        self.kernel_sizes = kernel_sizes or [11]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print('[i] reflection sythesis model: {}'.format({
            'kernel_sizes': kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'low_gamma': low_gamma, 'high_gamma': high_gamma}))

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))

        B_ = np.asarray(B, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.

        kernel_size = np.random.choice(self.kernel_sizes)
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)
        R_blur = R_
        kernel = cv2.getGaussianKernel(11, sigma)
        kernel2d = np.dot(kernel, kernel.T)

        for i in range(3):
            R_blur[..., i] = convolve2d(R_blur[..., i], kernel2d, mode='same')

        M_ = B_ + R_blur

        if np.max(M_) > 1:
            m = M_[M_ > 1]
            m = (np.mean(m) - 1) * gamma
            R_blur = np.clip(R_blur - m, 0, 1)
            M_ = np.clip(R_blur + B_, 0, 1)

        return B_, R_blur, M_

    
def _is_pil_image(img):
    """Determine whether the input is a PIL image or not."""
    return isinstance(img, Image.Image)


def mix_images(path1, path2, save_dir):
    """Mixes images from path1 and path2 and saves them to save_dir."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_names1 = os.listdir(path1)
    file_names2 = os.listdir(path2)

    for file_name in file_names1:
        if file_name not in file_names2:
            continue

        img1 = Image.open(os.path.join(path1, file_name))
        img2 = Image.open(os.path.join(path2, file_name))

        synth = ReflectionSythesis_1()
        B, R, T = synth(img1, img2)

        imgB = Image.fromarray(np.uint8(B*255))
        imgR = Image.fromarray(np.uint8(R*255))
        imgT = Image.fromarray(np.uint8(T*255))

        mixed = Image.blend(imgT, imgR, 0.3) # should be 0.3

        mixed.save(os.path.join(save_dir, file_name))
        print('Mixed image saved:', file_name)


if __name__ == '__main__':
    path1 = 'D:/Code/DereflectFormer-main/datasets/data_generation/synthetic/transmission_layer'
    path2 = 'D:/Code/DereflectFormer-main/datasets/data_generation/synthetic/reflection_layer'
    save_dir = 'D:/Code/DereflectFormer-main/datasets/data_generation/synthetic/mix_0.3'
    mix_images(path1, path2, save_dir)
