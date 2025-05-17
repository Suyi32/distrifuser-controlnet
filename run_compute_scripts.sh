python scripts/compute_metrics.py \
    --input_root0 /home/slida/DF-Serving/baselines/diffusers_controlnets_1_loras_1 \
    --input_root1 ./images_sdxl_t2i

python scripts/compute_metrics.py \
    --input_root0 /home/slida/DF-Serving/baselines/diffusers_controlnets_1_loras_1 \
    --input_root1 ./distrifuser_images/all_images_distrifuser_controlnets_1_loras_1_wrold_8_mode_corrected_async_gn

python scripts/compute_metrics.py \
    --input_root0 /home/slida/DF-Serving/baselines/diffusers_controlnets_1_loras_1 \
    --input_root1 /home/slida/DF-Serving/baselines/ours_nvlink_1_sync_loras_async_1


python scripts/compute_metrics.py \
    --input_root0 /home/slida/DF-Serving/baselines/diffusers_controlnets_1_loras_1 \
    --input_root1 /home/slida/DF-Serving/baselines/nirvana_controlnets_1_loras_1_skip_10


python scripts/compute_metrics.py \
    --input_root0 /home/slida/DF-Serving/baselines/diffusers_controlnets_1_loras_1 \
    --input_root1 /home/slida/DF-Serving/baselines/nirvana_controlnets_1_loras_1_skip_20