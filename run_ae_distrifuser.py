import os
import subprocess
import time
import re
import sys
import yaml
import json
import torch
import argparse

def cleanup(all_procs):
    print("Cleaning up...")
    # kill all
    for p in all_procs:
        print("kill", p.pid)
        # maybe we need kill using sigkill?
        os.system(f"kill -TERM {p.pid} > /dev/null 2>&1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_controlnets", type=int, required=True)
    parser.add_argument("--num_loras", type=int, required=True)
    args = parser.parse_args()

    setting = "{}C/{}L".format(args.num_controlnets, args.num_loras)
    logFolder = "/home/ubuntu/distrifuser-controlnet/distrifuser_benchmark_logs"
    if not os.path.exists(logFolder):
        os.makedirs(logFolder)

    
    commands = {
        "0C/0L": f"torchrun --nproc_per_node=2 run_distrifuser_benchmark.py --num_controlnets 0 --num_loras 0 > {logFolder}/distrifuser_0C_0L.log",
        "1C/0L": f"torchrun --nproc_per_node=4 run_distrifuser_benchmark.py --num_controlnets 1 --num_loras 0 > {logFolder}/distrifuser_1C_0L.log",
        "0C/1L": f"torchrun --nproc_per_node=2 run_distrifuser_benchmark.py --num_controlnets 0 --num_loras 1 > {logFolder}/distrifuser_0C_1L.log",
        "1C/1L": f"torchrun --nproc_per_node=4 run_distrifuser_benchmark.py --num_controlnets 1 --num_loras 1 > {logFolder}/distrifuser_1C_1L.log",
        "2C/2L": f"torchrun --nproc_per_node=8 run_distrifuser_benchmark.py --num_controlnets 2 --num_loras 2 > {logFolder}/distrifuser_2C_2L.log",
        "3C/2L": f"torchrun --nproc_per_node=8 run_distrifuser_benchmark.py --num_controlnets 3 --num_loras 2 > {logFolder}/distrifuser_3C_2L.log",
    }
    assert setting in commands, "Setting {} not found".format(setting)

    all_procs = []
    try:
        test_command = commands[setting]
        p = subprocess.Popen(test_command, shell=True)
        all_procs += [p]
    
        all_procs[0].wait()
        print("All processes finished")

    except Exception as e:
        print("Error:", e)
    finally:
        cleanup(all_procs)

if __name__ == '__main__':
    main()