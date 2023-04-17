#./tools/dist_test.sh configs/detection/self_enhance_detector/urpc/atss_r50_no_loss.py       /home/dong/BigDongDATA/self_enhance/gather_model/urpc/urpc_atss_self_enhance_new/epoch_12.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/urpc/faster_r50_no_loss.py     /home/dong/BigDongDATA/self_enhance/gather_model/urpc/urpc_faster_self_enhancement/epoch_12.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/urpc/retinanet_r50_no_loss.py  /home/dong/BigDongDATA/self_enhance/gather_model/urpc/urpc_retinanet_self_enhance/epoch_12.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/urpc/tood_r50_no_loss.py       /home/dong/BigDongDATA/self_enhance/gather_model/urpc/urpc_tood_self_enhance/epoch_12.pth 2
#
#
#./tools/dist_test.sh configs/detection/self_enhance_detector/rtts/atss_r50_no_loss.py /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtss_atss/epoch_12.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/rtts/faster_r50_no_loss.py /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtss_faster/epoch_12.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/rtts/retinanet_r50_no_loss.py /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtss_retinanet/epoch_12.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/rtts/tood_r50_no_loss.py /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtss_tood/epoch_12.pth 2

#./tools/dist_test.sh configs/detection/self_enhance_detector/exp3_voc/faster_with_strucloss.py /home/dong/BigDongDATA/self_enhance/gather_model/voc/voc_faster_with_strucloss2/epoch_4.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/exp3_voc/retina_with_strucloss.py /home/dong/BigDongDATA/self_enhance/gather_model/voc/voc_retina_with_strucloss_new1_4x4/epoch_4.pth 2



#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/atss_fpn_1x_rtts.py                  /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/atts_base/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/atss_fpn_1x_urpc2020.py              /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/atts_base/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/faster_rcnn_r50_fpn_1x_rtts.py       /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/faster_base/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/faster_rcnn_r50_fpn_1x_urpc2020.py   /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/faster_base/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/retinanet_r50_fpn_1x_rtts.py         /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/retina_base/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/retinanet_r50_fpn_1x_urpc2020.py     /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/retina_base/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/tood_fpn_1x_rtts.py                  /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/tood_base/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/self_enhance_detector/base_detector/tood_fpn_1x_urpc2020.py              /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/tood_base/epoch_12.pth     2


#./tools/dist_test.sh configs/detection/self_enhance_detector/exp3_voc/base_faster.py /home/dong/BigDongDATA/self_enhance/gather_model/voc/base_faster/epoch_4.pth 2
#./tools/dist_test.sh configs/detection/self_enhance_detector/exp3_voc/base_retina.py /home/dong/BigDongDATA/self_enhance/gather_model/voc/base_retina/epoch_4.pth 2
#
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/atss_fpn_1x_rtts.py              --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/atts_base/epoch_12.pth       --repeat-num 1
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/faster_rcnn_r50_fpn_1x_rtts.py   --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/faster_base/epoch_12.pth     --repeat-num 1
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/retinanet_r50_fpn_1x_rtts.py     --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/retina_base/epoch_12.pth     --repeat-num 1
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/tood_fpn_1x_rtts.py              --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/rtts/tood_base/epoch_12.pth       --repeat-num 1
#
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/atss_fpn_1x_urpc2020.py              --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/atts_base/epoch_12.pth   --repeat-num 5
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/faster_rcnn_r50_fpn_1x_urpc2020.py   --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/faster_base/epoch_12.pth --repeat-num 5
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/retinanet_r50_fpn_1x_urpc2020.py     --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/retina_base/epoch_12.pth --repeat-num 5
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/base_detector/tood_fpn_1x_urpc2020.py              --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/baseline/urpc/tood_base/epoch_12.pth   --repeat-num 5


#./tools/dist_test.sh configs/detection/aenet/urpc/aenet_atss_ufpn_1x_urpc2020.py   /home/dong/BigDongDATA/cycle_det/aenet/urpc/atts/randomseed/epoch_12.pth       2
#./tools/dist_test.sh configs/detection/aenet/urpc/aenet_faster_ufpn_1x_urpc2020.py /home/dong/BigDongDATA/cycle_det/aenet/urpc/faster/randomseed/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/aenet/urpc/aenet_retina_ufpn_1x_urpc2020.py /home/dong/BigDongDATA/cycle_det/aenet/urpc/retina/randomseed/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/aenet/urpc/aenet_tood_ufpn_1x_urpc2020.py   /home/dong/BigDongDATA/cycle_det/aenet/urpc/tood/randomseed_2/epoch_12.pth     2

#
./tools/dist_test.sh configs/detection/aenet/rtts/aenet_atss_ufpn_1x_rtts.py     /home/dong/BigDongDATA/cycle_det/aenet/rtts/atss/randomseed_2/epoch_12.pth       2
./tools/dist_test.sh configs/detection/aenet/rtts/aenet_faster_ufpn_1x_rtts.py   /home/dong/BigDongDATA/cycle_det/aenet/rtts/faster/randomseed/epoch_12.pth     2
#./tools/dist_test.sh configs/detection/aenet/rtts/aenet_retina_ufpn_1x_rtts.py   /home/dong/BigDongDATA/cycle_det/aenet/rtts/retina/randomseed/epoch_12.pth     2
./tools/dist_test.sh configs/detection/aenet/rtts/aenet_tood_ufpn_1x_rtts.py     /home/dong/BigDongDATA/cycle_det/aenet/rtts/tood/randomseed_3/epoch_12.pth     2


# python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/rtts/faster_r50_base_loss.py   --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtss_faster/epoch_12.pth     --repeat-num 1
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/rtts/retinanet_r50_base_loss.py   --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtts_retinanet/epoch_12.pth     --repeat-num 1
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/rtts/atss_r50_base_loss.py   --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtss_atss_new/epoch_12.pth     --repeat-num 1
#python tools/analysis_tools/benchmark.py configs/detection/self_enhance_detector/rtts/tood_r50_base_loss.py   --task inference --checkpoint /home/dong/BigDongDATA/self_enhance/gather_model/rtts/rtss_tood/epoch_12.pth     --repeat-num 1
#

#python tools/analysis_tools/benchmark.py configs/detection/aenet/rtts/aenet_faster_ufpn_1x_rtts.py   --task inference --checkpoint /home/dong/BigDongDATA/cycle_det/aenet/rtts/faster/randomseed/epoch_12.pth    --repeat-num 1
