from abc import ABCMeta, abstractmethod

import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from numpy import random

from lqit.registry import TRANSFORMS

try:
    from mmdet.structures.bbox import BaseBoxes  # noqa
    mmdet = True
except ImportError:
    mmdet = False


class FFTFilterBase(BaseTransform, metaclass=ABCMeta):

    def __init__(self, shape='cycle', choose=False, radius=16, get_gt=False):
        # TODO: support rhombus and soft cycle
        assert shape in ['cycle', 'square', 'r_square', 'cycle_band']
        self.shape = shape
        self.radius = radius
        self.choose = choose
        self.get_gt = get_gt
        self.fft_meta = dict()

    @property
    def center_mask(self):
        if isinstance(self.radius, int) or isinstance(self.radius, float):
            radius = self.radius
        elif isinstance(self.radius, list):
            if self.choose:
                radius = self.radius[random.randint(len(self.radius))]
            else:
                assert len(self.radius) == 2
                radius = random.uniform(self.radius[0], self.radius[1])
        else:
            raise ValueError

        self.fft_meta = dict()
        self.fft_meta['radius'] = radius
        self.fft_meta['shape'] = self.shape

        # avoid error
        radius = int(radius)

        if self.shape == 'cycle':
            mask = self._cycle_mask(radius=radius)
        elif self.shape == 'square':
            mask = self._square_mask(radius=radius)
        else:
            raise NotImplementedError
        return mask, radius

    def get_mask(self, results):
        if self.shape == 'cycle_band':
            mask = self._band_pass(results)
            return mask, None, None
        center_mask, radius = self.center_mask

        height, width = results['img_shape']
        x_c, y_c = width // 2, height // 2
        center_mask_y_c, center_mask_x_c = \
            center_mask.shape[0] // 2, center_mask.shape[1] // 2

        mask = np.zeros((height, width), dtype=np.bool)
        if y_c < radius:
            center_mask = center_mask[center_mask_y_c - y_c:center_mask_y_c +
                                      y_c + height % 2, :]
        if x_c < radius:
            center_mask = center_mask[:, center_mask_x_c -
                                      x_c:center_mask_x_c + x_c + width % 2]
        mask[max(y_c - radius, 0):y_c + radius,
             max(x_c - radius, 0):x_c + radius] = center_mask

        return mask, center_mask, radius

    def _band_pass(self, results):
        assert isinstance(self.radius, list) and len(self.radius) == 2
        radius_in, radius_out = int(self.radius[0]), int(self.radius[1])
        assert radius_in < radius_out

        self.fft_meta['shape'] = self.shape
        center_mask_in = self._cycle_mask(radius=radius_in)
        center_mask_out = self._cycle_mask(radius=radius_out)

        # in cycle
        height, width = results['img_shape']
        x_c, y_c = width // 2, height // 2
        center_mask_y_c, center_mask_x_c = \
            center_mask_in.shape[0] // 2, center_mask_in.shape[1] // 2

        mask_in = np.zeros((height, width), dtype=np.bool)
        if y_c < radius_in:
            center_mask_in = center_mask_in[(center_mask_y_c -
                                             y_c):(center_mask_y_c + y_c +
                                                   height % 2), :]
        if x_c < radius_in:
            center_mask_in = center_mask_in[:, (center_mask_x_c -
                                                x_c):(center_mask_x_c + x_c +
                                                      width % 2)]
        mask_in[max(y_c - radius_in, 0):y_c + radius_in,
                max(x_c - radius_in, 0):x_c + radius_in] = center_mask_in
        # out cycle
        center_mask_y_c, center_mask_x_c = \
            center_mask_out.shape[0] // 2, center_mask_out.shape[1] // 2

        mask_out = np.zeros((height, width), dtype=np.bool)
        if y_c < radius_out:
            center_mask_out = center_mask_out[(center_mask_y_c -
                                               y_c):(center_mask_y_c + y_c +
                                                     height % 2), :]
        if x_c < radius_out:
            center_mask_out = center_mask_out[:, (center_mask_x_c -
                                                  x_c):(center_mask_x_c + x_c +
                                                        width % 2)]
        mask_out[max(y_c - radius_out, 0):y_c + radius_out,
                 max(x_c - radius_out, 0):x_c + radius_out] = center_mask_out

        mask_out = ~mask_out
        if not hasattr(self, 'pass_type'):
            mask = mask_out + mask_in
        elif self.pass_type == 'low':
            mask = mask_out + mask_in
            mask = ~mask
        elif self.pass_type == 'high':
            mask = mask_out + mask_in
        else:
            raise KeyError
        return mask

    def _cycle_mask(self, radius):
        y, x = np.ogrid[:2 * radius, :2 * radius]
        cycle_mask = ((x - radius)**2 + (y - radius)**2) <= \
                     (radius - 1)**2
        return cycle_mask

    def _square_mask(self, radius):
        square_mask = np.ones((radius * 2, radius * 2), dtype=np.bool)
        return square_mask

    def fft(self, channel_img):
        # Fourier transform
        f = np.fft.fft2(channel_img)
        # Shift the spectrum to the central location
        fshift = np.fft.fftshift(f)
        return fshift

    def ifft(self, fshift):
        # Shift the spectrum to its original location
        ishift = np.fft.ifftshift(fshift)
        # Inverse Fourier Transform
        iimg = np.fft.ifft2(ishift)
        # kep the input and output have same type
        iimg = np.abs(iimg).astype(np.float32)
        # avoid noise
        iimg = np.clip(iimg, a_min=0, a_max=255)
        return iimg

    @abstractmethod
    def transform(self, results):
        """The transform function.

        All subclass of BaseTransform should override this method.
        """


@TRANSFORMS.register_module()
class FFTFilter(FFTFilterBase):
    """

    choose: select one of the radius in the list
    """

    def __init__(self,
                 pass_type='low',
                 prob=None,
                 w_high=None,
                 w_low=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert pass_type in ['low', 'high', 'random', 'none', 'soft']
        self.pass_type = pass_type
        if pass_type == 'random':
            assert prob is not None
        self.prob = prob
        if self.pass_type == 'soft':
            assert w_low is not None and w_high is not None
        if isinstance(w_low, list) or isinstance(w_low, tuple):
            assert len(w_low) == 2
        if isinstance(w_high, list) or isinstance(w_high, tuple):
            assert len(w_high) == 2
        self.w_high = w_high
        self.w_low = w_low

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    def fft_filter(self, img, mask):
        if len(img.shape) == 3:
            channel = img.shape[-1]
        elif len(img.shape) == 2:
            channel = 1
        else:
            raise ValueError

        result_img = np.empty_like(img, dtype=np.int64)
        for i in range(channel):
            fshift = self.fft(channel_img=img[:, :, i])
            filter_fshift = mask * fshift
            iimg = self.ifft(fshift=filter_fshift)
            result_img[:, :, i] = iimg
        return result_img

    def get_gt_frequency(self, img):
        if len(img.shape) == 3:
            channel = img.shape[-1]
        elif len(img.shape) == 2:
            channel = 1
        else:
            raise ValueError

        result_img = np.empty_like(img, dtype=np.complex128)
        for i in range(channel):
            fshift = self.fft(channel_img=img[:, :, i])
            result_img[:, :, i] = fshift
        return result_img

    def get_soft_mask(self, results):
        center_mask, radius = self.center_mask

        height, width = results['img_shape']
        x_c, y_c = width // 2, height // 2
        center_mask_y_c, center_mask_x_c = \
            center_mask.shape[0] // 2, center_mask.shape[1] // 2

        mask_low = np.zeros((height, width), dtype=np.bool)
        if y_c < radius:
            center_mask = center_mask[center_mask_y_c - y_c:center_mask_y_c +
                                      y_c + height % 2, :]
        if x_c < radius:
            center_mask = center_mask[:, center_mask_x_c -
                                      x_c:center_mask_x_c + x_c + width % 2]
        mask_low[max(y_c - radius, 0):y_c + radius,
                 max(x_c - radius, 0):x_c + radius] = center_mask

        mask_high = ~mask_low
        if isinstance(self.w_low, list) or isinstance(self.w_low, tuple):
            w_low = random.uniform(self.w_low[0], self.w_low[1])
        else:
            w_low = self.w_low
        if isinstance(self.w_high, list) or isinstance(self.w_high, tuple):
            w_high = random.uniform(self.w_high[0], self.w_high[1])
        else:
            w_high = self.w_high
        self.fft_meta['w_high'] = w_high
        self.fft_meta['w_low'] = w_low
        mask = mask_high * w_high + mask_low * w_low
        self.fft_meta['weight_mask'] = mask
        return mask, center_mask, radius

    def transform(self, results: dict):
        img = results['img']
        pass_type = self.pass_type
        if pass_type == 'none':
            results['fft_filter_img'] = img
            return results
        elif pass_type == 'soft':
            results['fft_meta'] = self.fft_meta
            mask, _, _ = self.get_soft_mask(results)
        else:
            mask, _, _ = self.get_mask(results)

            if pass_type == 'high' or \
                    (pass_type == 'random' and
                     self._random_prob() < self.prob):
                mask = ~mask
                pass_type = 'high'

        self.fft_meta['pass_type'] = pass_type
        results['fft_meta'] = self.fft_meta
        result_img = self.fft_filter(img=img, mask=mask)
        results['fft_filter_img'] = result_img
        return results


@TRANSFORMS.register_module()
class AddHighPassImg(FFTFilter):

    def __init__(self, *args, **kwargs):
        pass_type = 'high'
        super().__init__(*args, pass_type=pass_type, **kwargs)

    def transform(self, results: dict):
        assert self.pass_type == 'high'
        img = results['img']
        mask, _, _ = self.get_mask(results)
        mask = ~mask

        high_pass_img = self.fft_filter(img=img, mask=mask)
        result_img = img + high_pass_img
        result_img = np.clip(result_img, a_min=0, a_max=255).astype(np.uint8)
        results['img'] = result_img
        return results
