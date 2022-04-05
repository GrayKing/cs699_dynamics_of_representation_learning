python -W ignore main.py --result_folder "results_img_nsf/both"  --loss_type both --method nsf --num_samples=20000 --nsteps 1000 --stepsize 0.0005 --nkernels=5 --density image

python -W ignore main.py --result_folder "results_img_nsf/both"  --loss_type both --method nsf --num_samples=20000 --nsteps 2000 --stepsize 0.0005 --nkernels=5 --density image

python -W ignore main.py --result_folder "results_img_nsf/both"  --loss_type both --method nsf --num_samples=20000 --nsteps 4000 --stepsize 0.0005 --nkernels=5 --density image

(csci699) Tianchengs-MBP:sampling tianchengjin$ sh experiment_both_nsf_image 
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.25624436806647893
TV distance is 0.7353355462031538
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.2323244904598254
TV distance is 0.708440051758721
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
KL divergence is 0.15456199103901816
TV distance is 0.6669332009752658