import subprocess


size = 128
split = size

for pre_core in [5, 10, 20]:
    cmd1 = f'python ./run_vortex_llg.py --w {size} \
                                        --layers 2 \
                                        --krn 16 \
                                        --split {split} \
                                        --pre_core {pre_core} \
                                        --error_min 1.0e-5 \
                                        --dtime 5.0e-13 \
                                        --max_iter 80000 \
                                        --nsamples 30'
    subprocess.run(cmd1, shell=True)


    cmd2 = f'python ./analyze_vortex.py --w {size} \
                                        --krn 16 \
                                        --split {split} \
                                        --pre_core {pre_core} \
                                        --errorfilter 1e-5'
    subprocess.run(cmd2, shell=True)



'''
python ./analyze_vortex.py --w 128 --krn 16 --split 128 --pre_core 5 --errorfilter 1e-5
'''