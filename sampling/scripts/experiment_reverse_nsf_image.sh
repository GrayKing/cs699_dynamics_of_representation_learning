python -W ignore main.py --result_folder "results_img_nsf/reverse"  --loss_type reverse --method nsf --num_samples=20000 --nsteps 1000 --stepsize 0.0001 --nkernels=5 --density image

python -W ignore main.py --result_folder "results_img_nsf/reverse"  --loss_type reverse --method nsf --num_samples=20000 --nsteps 2000 --stepsize 0.0001 --nkernels=5 --density image

python -W ignore main.py --result_folder "results_img_nsf/reverse"  --loss_type reverse --method nsf --num_samples=20000 --nsteps 4000 --stepsize 0.0001 --nkernels=5 --density image

WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.35639933323251627
TV distance is 0.7708090082540997
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.2903146524978312
TV distance is 0.7364280370152538
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.26712521713885634
TV distance is 0.7285519024856216

