GPUNUM=1
PROCESSNUM=1
PART=basemodel
JOBNAME=install

srun --partition=${PART} -n${PROCESSNUM} --gres=gpu:${GPUNUM} --job-name=${JOBNAME} --ntasks-per-node=${GPUNUM} --cpus-per-task=5 --quotatype=auto --kill-on-bad-exit=1 \
    python setup.py build install