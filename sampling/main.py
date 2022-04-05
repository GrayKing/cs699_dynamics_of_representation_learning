"""demonstrating some utilties in the starter code"""
import argparse
import os

import jax
import matplotlib.image
import matplotlib.pyplot
import numpy 

import NPEET.npeet.entropy_estimators
from utils.metrics import get_discretized_tv_for_image_density
from utils.density import continuous_energy_from_image, prepare_image, continuous_energy_from_image_ext
from utils.density import sample_from_image_density, sample_from_normal
from utils.density import gaussian_mixture_energy, gaussian_mixture_sampler
from utils.sampler_hmc import sample_hmc
from utils.sampler_mh import sample_mh
from utils.sampler_nsf import generate
#from utils.sampler_nut import sample_nut 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="results")
    parser.add_argument("--method", type=str, choices=["mh","hmc","nsf"], default="nsf")
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.1)
    parser.add_argument("--nkernels", type=int, default=5)
    parser.add_argument("--factor", type=int, default=100)
    parser.add_argument("--density", type=str, choices=['image','mog'],default='image')
    parser.add_argument("--num_samples",type=int, default=20000)
    parser.add_argument("--make_anime",type=bool,default=False)
    parser.add_argument("--loss_type",type=str, 
        choices=["forward","reverse","both"],default="forward")

    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    os.makedirs(f"{args.result_folder}", exist_ok=True)

    num_samples = args.num_samples

    if args.density == "image":
        # load some image
        img = matplotlib.image.imread('./data/labrador.jpg')

        # convert to energy function
        # first we get discrete energy and density values
        density, energy = prepare_image(
            img, crop=(10, 710, 240, 940), white_cutoff=225, gauss_sigma=5, background=0.01
        )

        scale = 100.0 
        mean = 350.0

        fig = matplotlib.pyplot.figure(figsize=(10, 10))
        
        x_max, y_max = density.shape
        xp = jax.numpy.arange(x_max)
        yp = jax.numpy.arange(y_max)
        zp = jax.numpy.array(density)

        # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
        energy_fn = lambda coord: continuous_energy_from_image_ext(coord, xp, yp, zp, fill_value=10)
        energy_total_fn = lambda coord: jax.numpy.sum(continuous_energy_from_image_ext(coord, xp, yp, zp, fill_value=10))
        energy_fn_grad = jax.grad(energy_total_fn)

        # prepare data 
        key, subkey = jax.random.split(key)
        data = sample_from_image_density(args.num_samples, density, subkey)

        # generate another set of samples from true distribution, to demonstrate comparisons
        key, subkey = jax.random.split(key)
        second_samples = sample_from_image_density(args.num_samples, density, subkey)

    elif args.density == "mog":

        mog_means = numpy.array([[0,0],[2,0],[-2,0]])
        mog_sigmas = numpy.array([5.0,numpy.sqrt(0.1),numpy.sqrt(0.1)])
        mog_weights = numpy.array([1e-5,0.5-(5e-6),0.5-(5e-6)])

        key, subkey = jax.random.split(key)
        data = gaussian_mixture_sampler(num_samples, 
            means = mog_means,
            sigmas = mog_sigmas ,
            weights = mog_weights,
            key = subkey)

        key, subkey = jax.random.split(key)
        second_samples = gaussian_mixture_sampler(num_samples, 
            means = mog_means,
            sigmas = mog_sigmas ,
            weights = mog_weights,
            key = subkey)

        # no scalling 
        scale = 1.0 
        mean = 0.0 

        # prepare energy function 
        energy_fn = lambda coord: gaussian_mixture_energy(coord, mog_means, mog_sigmas, mog_weights)
        energy_total_fn = lambda coord: jax.numpy.sum(gaussian_mixture_energy(coord, mog_means, mog_sigmas, mog_weights))
        energy_fn_grad = jax.grad(energy_total_fn)

        # perpare density image 
        x,y = numpy.meshgrid(numpy.arange(-5,5, 0.1), numpy.arange(-5,5, 0.1))
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        pts = numpy.hstack([y,x])
        density = numpy.exp(-energy_fn(pts))
        density = density.reshape(100,100)
    else:
        pass 


    if args.method == "hmc":
        # generate samples from true distribution
        key, subkey = jax.random.split(key)
        factor = args.factor
        data = sample_from_normal(num_samples//factor,subkey) * scale + mean 
        collected_data = []
        
        for i in range(factor):
            key, subkey = jax.random.split(key)
            samples, samples_dW = sample_hmc(data,key,
                energy_fn=energy_fn, 
                energy_fn_grad=energy_fn_grad,
                nsteps=args.nsteps, 
                stepsize=args.stepsize)
            data = samples
            collected_data.append(data.copy())
        
        samples = numpy.vstack(collected_data)
    elif args.method == "nsf":
        samples = generate(data,
                args,
                scale=scale,
                mean=mean,
                nkernels=args.nkernels,
                nsteps=args.nsteps,
                energy_fn=energy_fn,
                density=density,
                stepsize=args.stepsize)
    elif args.method == "mh":
        pass 

    # (scatter) plot the samples with image in background
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(numpy.array(samples)[:, 1], numpy.array(samples)[:, 0], s=0.5, alpha=0.5)
    
    if args.density == "image":
        ax.imshow(density, alpha=0.3)
    else:
        ax.axis('equal')
        ax.imshow(density, extent=[-5, 5, -5, 5], alpha=0.3)
        ax.set(xlim=(-5,5), ylim=(-5, 5))

    if args.method == "hmc":
        fig.savefig(f"{args.result_folder}/n{args.nsteps}.png")
    elif args.method == "nsf":
        fig.savefig(f"{args.result_folder}/n{args.nsteps}_k{args.nkernels}.png")
    else:
        pass 

    # We have samples from two distributions. We use NPEET package to compute kldiv directly from samples.
    # NPEET needs nxd tensors
    kldiv = NPEET.npeet.entropy_estimators.kldiv(samples, second_samples)
    print(f"KL divergence is {kldiv}")

    # TV distance between discretized density
    # The discrete density bin from the image give us a natural scale for discretization.
    # We compute discrete density from sample at this scale and compute the TV distance between the two densities
    if args.density == "image":
        tv_dist = get_discretized_tv_for_image_density(
            numpy.asarray(density), numpy.asarray(samples), bin_size=[7, 7]
        )
        print(f"TV distance is {tv_dist}")
    else:
        pass 
