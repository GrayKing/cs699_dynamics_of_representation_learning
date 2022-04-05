from typing import Optional

import jax.numpy
import jax.numpy as np 
from jax import grad,jit,vmap 

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from jax.experimental import stax # neural network library
from jax.experimental.stax import Dense, Relu # neural network layers

from jax.experimental import optimizers
from jax import jit, grad
import numpy as onp

rng = jax.random.PRNGKey(0)

def sample_normal(N):
  D = 2
  return jax.random.normal(rng, (N, D))

def log_prob_normal(x):
  return np.sum(-np.square(x)/2.0,axis=-1) + 0.39908993417

def nvp_forward(net_params, shift_and_log_scale_fn, x, flip=False):
  d = x.shape[-1]//2
  x1, x2 = x[:, :d], x[:, d:]
  if flip:
    x2, x1 = x1, x2
  shift, log_scale = shift_and_log_scale_fn(net_params, x1)
  y2 = x2*np.exp(log_scale) + shift
  if flip:
    x1, y2 = y2, x1
  y = np.concatenate([x1, y2], axis=-1)
  return y, log_scale 

def nvp_inverse(net_params, 
  shift_and_log_scale_fn, 
  y, 
  flip=False):
  d = y.shape[-1]//2
  y1, y2 = y[:, :d], y[:, d:]
  if flip:
    y1, y2 = y2, y1
  shift, log_scale = shift_and_log_scale_fn(net_params, y1)
  x2 = (y2-shift)*np.exp(-log_scale)
  if flip:
    y1, x2 = x2, y1
  x = np.concatenate([y1, x2], axis=-1)
  return x, log_scale

def sample_nvp(net_params, shift_log_scale_fn, base_sample_fn, N, flip=False):
  x = base_sample_fn(N)
  return nvp_forward(net_params, shift_log_scale_fn, x, flip=flip)

def log_prob_nvp(net_params, shift_log_scale_fn, base_log_prob_fn, y, flip=False):
  x, log_scale = nvp_inverse(net_params, shift_log_scale_fn, y, flip=flip)
  ildj = -np.sum(log_scale, axis=-1)
  return base_log_prob_fn(x) + ildj

def init_nvp():
  D = 2
  net_init, net_apply = stax.serial(
    Dense(64), 
    Relu, 
    Dense(64), 
    Relu, 
    Dense(64),
    Relu,
    Dense(D)
  )
  in_shape = (-1, D//2)
  out_shape, net_params = net_init(rng, in_shape)
  def shift_and_log_scale_fn(net_params, x1):
    s = net_apply(net_params, x1)
    return np.split(s, 2, axis=1)
  return net_params, shift_and_log_scale_fn

def init_nvp_chain(n=2):
  flip = False
  ps, configs = [], []
  for i in range(n):
    p, f = init_nvp()
    ps.append(p)
    configs.append((f, flip))
    flip = not flip
  return ps, configs

def make_log_prob_fn(p, log_prob_fn, config):
  shift_log_scale_fn, flip = config
  return lambda x: log_prob_nvp(p, shift_log_scale_fn, log_prob_fn, x, flip=flip)

def log_prob_nvp_chain(ps, configs, base_log_prob_fn, y):
  log_prob_fn = base_log_prob_fn
  for p, config in zip(ps, configs):
    log_prob_fn = make_log_prob_fn(p, log_prob_fn, config)
  return log_prob_fn(y)

def energy_nvp_chain(ps, configs, base_energy_fn, x):
    energy_fn = base_energy_fn
    tmp_loss = 0 
    y = x 
    for p, config in zip(ps, configs):
        shift_log_scale_fn, flip = config
        y, jacob_loss = nvp_forward(p, shift_log_scale_fn, y, flip=flip)
        tmp_loss = tmp_loss + np.sum(jacob_loss,axis=-1) 
    return energy_fn(y) - tmp_loss

def generate(
    X,
    args,
    nkernels=5,
    scale=1.0,
    mean=0.0,
    log_prob=log_prob_normal,
    nsteps=2e3,
    stepsize=1e-3,
    train_energy=False,
    density=None,
    energy_fn=None): 
    ps, cs = init_nvp_chain(nkernels)
    
    tgt_energy_fn = lambda coord: energy_fn(mean+(coord*scale))

    @jit 
    def loss_foward(params, batch):
      return -np.mean(log_prob_nvp_chain(params, cs, log_prob, batch))

    @jit 
    def loss_reverse(params, batch):
      tmp_loss = energy_nvp_chain(params, cs, tgt_energy_fn, batch)
      cond = jax.numpy.isnan(tmp_loss)
      loss = jax.numpy.where(cond, 0, tmp_loss)
      cnt =  loss.shape[0] - jax.numpy.sum(cond)
      return np.mean(loss) * cnt / loss.shape[0]

    opt_init, opt_update, get_params = optimizers.adam(step_size=stepsize)

    @jit
    def step(i, opt_state, batch1, batch2):
      params = get_params(opt_state)

      loss_F = lambda x,y:0 
      if args.loss_type == "forward" or args.loss_type == "both":
        loss_F = lambda x,y:loss_foward(x,y)

      loss_R = lambda x,y:0 
      if args.loss_type == "reverse" or args.loss_type == "both":
        loss_R = lambda x,y:loss_reverse(x,y) 

      loss = lambda params,batch1,batch2: loss_F(params,batch1) + loss_R(params,batch2)
      g = grad(loss)(params, batch1, batch2)
      return loss_F(params,batch1), loss_R(params,batch2), opt_update(i, g, opt_state)

    iters = int(nsteps)
    opt_state = opt_init(ps)

    data_scale = scale 
    data_mean = mean 
    X = (X - data_mean) / data_scale
    data_generator_1 = (X[onp.random.choice(X.shape[0], 250)] for _ in range(iters))

    U = sample_normal(X.shape[0])
    data_generator_2 = (U[onp.random.choice(U.shape[0], 250)] for _ in range(iters))

    losses_F = []
    losses_R = [] 

    for i in range(iters):
      loss_F, loss_R, opt_state = step(i, opt_state, next(data_generator_1),next(data_generator_2))
      losses_F.append(loss_F)
      losses_R.append(loss_R)
      if (i%100==0):
        #print("Step %d: "%(i),onp.mean(losses_F),onp.mean(losses_R))
        pass
    ps = get_params(opt_state)

    x = sample_normal(args.num_samples)
    values = [x*data_scale+data_mean]
    for p, config in zip(ps, cs):
      shift_log_scale_fn, flip = config
      x, _ = nvp_forward(p, shift_log_scale_fn, x, flip=flip)
      values.append(x*data_scale+data_mean)

    fig, ax = plt.subplots(figsize=(10, 10))
    losses_F = [ losses_F[i] for i in range(0,len(losses_F),20) ]
    losses_R = [ losses_R[i] for i in range(0,len(losses_R),20) ]
    ax.plot(onp.arange(len(losses_R)),losses_R, label='Reverse KL-divergence')
    ax.plot(onp.arange(len(losses_F)),losses_F, label='Forward KL-divergence')
    ax.legend(loc='upper left')
    fig.savefig(f"{args.result_folder}/n{args.nsteps}_k{args.nkernels}_result.png")
    plt.clf()

    # First set up the figure, the axis, and the plot element we want to animate
    if args.make_anime is True:
        from matplotlib import animation, rc
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('equal')
        if args.density == "image": 
            ax.imshow(density, alpha=0.3)
            #ax.set(xlim=(0, 700), ylim=(0, 700))
        else:
            ax.set(xlim=(-5, 5), ylim=(-5, 5))
            ax.imshow(density, extent=[-5, 5, -5, 5], alpha=0.3)

        y = values[0]
        paths = ax.scatter(y[:, 1], y[:, 0], s=0.5, alpha=0.5)
        values.append(values[-1])

        def animate(i):
            l = int(i)//48
            t = (float(i%48))/48
            y = (1-t)*values[l] + t*values[l+1]
            paths.set_offsets(y[:,[1,0]])
            return (paths,)

        anim = animation.FuncAnimation(fig, animate, frames=48*len(cs)+48, interval=1)
        anim.save(f"{args.result_folder}/anim.mp4",fps=60)

    return values[-1]


if __name__ == "__main__":
    n_samples = 2000
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, y = noisy_moons
    X = StandardScaler().fit_transform(X)


