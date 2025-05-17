torchrun --nproc_per_node=8 run_distrifuser_gen_images.py --num_controlnets 1 --num_loras 2
torchrun --nproc_per_node=8 run_distrifuser_gen_images.py --num_controlnets 1 --num_loras 1
scancel 162833
