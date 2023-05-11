from typing import List, Optional, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from numpy import ndarray

from lqit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GetEdgeGTFromImage(BaseTransform):
    """Get the edge from image, and set it into results dict.

    Required Keys:

        - img

    Modified Keys:

        - edge_img

    Args:
        method (str): The calculate edge method. Defaults to 'scharr'.
        kernel_size: (List[int] or int) The gaussian blur kernel size.
            Defaults to [3, 3].
        threshold_value (int) The threshold value which used in 'roberts',
            'prewitt', 'sobel', and 'laplacian'. Defaults to 127.
        results_key (str): The name that going to save gt image in the results
            dict. Defaults to 'gt_edge'.

    Note:
        This transforms should add before `PackInputs`. Otherwise, some
        transforms will change the `img` and do not change `gt_edge`.
    """

    def __init__(self,
                 method: str = 'scharr',
                 kernel_size: Union[List[int], int] = [3, 3],
                 threshold_value: int = 127,
                 results_key: str = 'gt_edge') -> None:
        assert method in [
            'roberts', 'prewitt', 'canny', 'scharr', 'sobel', 'laplacian',
            'log'
        ]
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, (tuple, list)):
            self.kernel_size = tuple(kernel_size)
        else:
            raise TypeError('kernel_size should be a list of int or int,'
                            f'but get {type(kernel_size)}')
        self.threshold_value = threshold_value
        self.method = method
        self.results_key = results_key

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to get edge image from image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert 'img' in results
        img = results['img']
        get_edge_func = getattr(self, self.method)
        edge_img = get_edge_func(img)
        results[self.results_key] = edge_img
        return results

    def _get_gaussian_blur(self,
                           img: ndarray,
                           threshold: bool = True) -> ndarray:
        """Get gaussian blur of the image.

        Args:
            img (np.ndarry): The image going to be blurred by gaussian.
            threshold (bool): Whether to threshold the blurred image.

        Returns:
            np.ndarry: The blurred image or the threshold image.
        """
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray_image, self.kernel_size, 0)
        if threshold:
            _, binary = cv2.threshold(gaussian_blur, self.threshold_value, 255,
                                      cv2.THRESH_BINARY)
            return binary
        else:
            return gaussian_blur

    def roberts(self, img: ndarray) -> ndarray:
        """Get image based on roberts.

        Args:
            img (np.ndarry): The image going to get edge image.

        Returns:
            np.ndarry: The edge image.
        """
        binary = self._get_gaussian_blur(img=img, threshold=True)

        kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(binary, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(binary, cv2.CV_16S, kernel_y)
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        return edge

    def prewitt(self, img: ndarray) -> ndarray:
        """Get image based on prewitt.

        Args:
            img (np.ndarry): The image going to get edge image.

        Returns:
            np.ndarry: The edge image.
        """
        binary = self._get_gaussian_blur(img=img, threshold=True)

        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(binary, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(binary, cv2.CV_16S, kernel_y)
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        return edge

    def sobel(self, img: ndarray) -> ndarray:
        """Get image based on sobel.

        Args:
            img (np.ndarry): The image going to get edge image.

        Returns:
            np.ndarry: The edge image.
        """
        binary = self._get_gaussian_blur(img=img, threshold=True)

        x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        return edge

    def laplacian(self, img: ndarray) -> ndarray:
        """Get image based on laplacian.

        Args:
            img (np.ndarry): The image going to get edge image.

        Returns:
            np.ndarry: The edge image.
        """
        binary = self._get_gaussian_blur(img=img, threshold=True)

        dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
        edge = cv2.convertScaleAbs(dst)
        return edge

    def scharr(self, img: ndarray) -> ndarray:
        """Get image based on scharr.

        Args:
            img (np.ndarry): The image going to get edge image.

        Returns:
            np.ndarry: The edge image.
        """
        gaussian_blur = self._get_gaussian_blur(img=img, threshold=False)

        x = cv2.Scharr(gaussian_blur, cv2.CV_32F, 1, 0)
        y = cv2.Scharr(gaussian_blur, cv2.CV_32F, 0, 1)
        abs_x = cv2.convertScaleAbs(x)
        abs_y = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        return edge

    def canny(self, img: ndarray) -> ndarray:
        """Get image based on canny.

        Args:
            img (np.ndarry): The image going to get edge image.

        Returns:
            np.ndarry: The edge image.
        """
        gaussian_blur = self._get_gaussian_blur(img=img, threshold=False)

        edge = cv2.Canny(gaussian_blur, 50, 150)
        return edge

    def log(self, img: ndarray) -> ndarray:
        """Get image based on log.

        Args:
            img (np.ndarry): The image going to get edge image.

        Returns:
            np.ndarry: The edge image.
        """
        gaussian_blur = self._get_gaussian_blur(img=img, threshold=False)

        dst = cv2.Laplacian(gaussian_blur, cv2.CV_16S, ksize=3)
        edge = cv2.convertScaleAbs(dst)
        return edge

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'method={self.method}, '
                    f'threshold_value={self.threshold_value}, '
                    f'method={self.method}, '
                    f"results_key='{self.results_key}')")
        return repr_str
