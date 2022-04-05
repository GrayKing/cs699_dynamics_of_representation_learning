python -W ignore main.py --result_folder "results_img_nsf/forward"  --loss_type forward --method nsf --num_samples=20000 --nsteps 1000 --stepsize 0.0005 --nkernels=5 --density image

python -W ignore main.py --result_folder "results_img_nsf/forward"  --loss_type forward --method nsf --num_samples=20000 --nsteps 2000 --stepsize 0.0005 --nkernels=5 --density image

python -W ignore main.py --result_folder "results_img_nsf/forward"  --loss_type forward --method nsf --num_samples=20000 --nsteps 4000 --stepsize 0.0005 --nkernels=5 --density image

WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.21129724210236073
TV distance is 0.7034038829240727
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.18115943964637532
TV distance is 0.6865954929174082
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.15754050863560964
TV distance is 0.6585662980192987

