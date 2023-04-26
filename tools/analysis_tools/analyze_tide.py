from lqit.detection.evaluation.tide import COCO, TIDE, COCOResult

out_path = 'work_dirs/tide/show/'
result_path = '/home/dong/BigDongDATA/uwdet_ckpt/gather_ckpt/gather_result/atss_r50_1x/result/14_atss_acdc.bbox.json'  # noqa
gt_path = '/home/dong/BigDongDATA/DATA/URPC/annotations_json/val.json'

tide = TIDE()
tide.evaluate(COCO(path=gt_path), COCOResult(path=result_path), mode='bbox')
tide.summarize()

errors = tide.get_all_errors()
tide.plot(out_dir=out_path)
