python -W ignore main.py --result_folder "results_mog_nsf/forward"  --loss_type forward --method nsf --num_samples=10000 --nsteps 1000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/forward"  --loss_type forward --method nsf --num_samples=10000 --nsteps 2000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/forward"  --loss_type forward --method nsf --num_samples=10000 --nsteps 4000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/both"  --loss_type both --method nsf --num_samples=10000 --nsteps 1000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/both"  --loss_type both --method nsf --num_samples=10000 --nsteps 2000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/both"  --loss_type both --method nsf --num_samples=10000 --nsteps 4000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/reverse"  --loss_type reverse --method nsf --num_samples=10000 --nsteps 1000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/reverse"  --loss_type reverse --method nsf --num_samples=10000 --nsteps 2000 --stepsize 0.0001 --nkernels=5 --density mog

python -W ignore main.py --result_folder "results_mog_nsf/reverse"  --loss_type reverse --method nsf --num_samples=10000 --nsteps 4000 --stepsize 0.0001 --nkernels=5 --density mog
