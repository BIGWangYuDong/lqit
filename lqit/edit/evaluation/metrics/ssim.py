from typing import Optional, Sequence

from mmeval.metrics import SSIM as MMEVAL_SSIM

from lqit.registry import METRICS


@METRICS.register_module()
class SSIM(MMEVAL_SSIM):
    """Calculate SSIM (structural similarity). A wrapper of :
    class:`mmeval.SSIM`.

    Ref:

    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        gt_key (str): Key of ground-truth. Defaults to 'img'
        pred_key (str): Key of prediction. Defaults to 'pred_img'
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Defaults to 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Defaults to 'CHW'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Defaults to None.
        channel_order (str): The channel order of image. Defaults to 'rgb'.
        scaling (float, optional): Scaling factor for final metric.
            E.g. scaling=100 means the final metric will be amplified by 100
            for output. Defaults to 1.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None
        dist_backend (str | None): The name of the distributed communication
            backend. Refer to :class:`mmeval.BaseMetric`.
            Defaults to 'torch_cuda'.

    Metrics:
        - SSIM (float): Structural similarity
    """

    metric = 'SSIM'

    def __init__(self,
                 gt_key: str = 'img',
                 pred_key: str = 'pred_img',
                 input_order='CHW',
                 crop_border=0,
                 convert_to=None,
                 channel_order: str = 'rgb',
                 prefix: Optional[str] = None,
                 scaling: float = 1,
                 dist_backend: str = 'torch_cuda',
                 **kwargs) -> None:
        super().__init__(
            crop_border=crop_border,
            input_order=input_order,
            convert_to=convert_to,
            channel_order=channel_order,
            dist_backend=dist_backend,
            **kwargs)

        self.gt_key = gt_key
        self.pred_key = pred_key
        self.prefix = prefix
        self.scaling = scaling

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        predictions, groundtruths = [], []
        for data_sample in data_samples:
            pred_img = data_sample['pred_pixel'].get(
                self.pred_key).cpu().numpy()
            assert data_sample.get('gt_pixel', None) is not None
            gt_img = data_sample['gt_pixel'].get(self.gt_key).cpu().numpy()
            predictions.append(pred_img)
            groundtruths.append(gt_img)
        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs):
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """

        metric_results = self.compute(*args, **kwargs)
        self.reset()

        key_template = f'{self.prefix}/{{}}' if self.prefix else '{}'
        return {
            key_template.format(k): v * self.scaling
            for k, v in metric_results.items()
        }
