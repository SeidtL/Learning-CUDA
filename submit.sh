export SKIP_TRACE=1 
srun --gres=gpu:nvidia:1 --cpus-per-task=16 --mem=16G make
