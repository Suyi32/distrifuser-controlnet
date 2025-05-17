logFolder="./distrifuser_benchmark_logs"

# # 0C/0L
# torchrun --nproc_per_node=2 run_distrifuser_benchmark.py --num_controlnets 0 --num_loras 0 > $logFolder/distrifuser_0C_0L.log
# # 1C/0L
# torchrun --nproc_per_node=4 run_distrifuser_benchmark.py --num_controlnets 1 --num_loras 0 > $logFolder/distrifuser_1C_0L.log
# # 0C/1L
torchrun --nproc_per_node=2 run_distrifuser_benchmark.py --num_controlnets 0 --num_loras 1 > $logFolder/distrifuser_0C_1L.log
# # 1C/1L
# torchrun --nproc_per_node=4 run_distrifuser_benchmark.py --num_controlnets 1 --num_loras 1 > $logFolder/distrifuser_1C_1L.log
# # 2C/2L
# torchrun --nproc_per_node=8 run_distrifuser_benchmark.py --num_controlnets 2 --num_loras 2 > $logFolder/distrifuser_2C_2L.log
# # # 3C/2L
# torchrun --nproc_per_node=8 run_distrifuser_benchmark.py --num_controlnets 3 --num_loras 2 > $logFolder/distrifuser_3C_2L.log