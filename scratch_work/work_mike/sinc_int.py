#! /usr/bin/python
#
# Author: Gaute Hope (gaute.hope@nersc.no) / 2015
#
# based on example from matlab sinc function and
# interpolate.m by H. Hobæk (1994).
#
# this implementation is similar to the matlab sinc-example, but
# calculates the values sequentially and not as a single matrix
# matrix operation for all the values.
#

import scipy as sc
import numpy as np

def resample (x, k):
  """
  Resample the signal to the given ratio using a sinc kernel

  input:

    x   a vector or matrix with a signal in each row
    k   ratio to resample to

    returns

    y   the up or downsampled signal

    when downsampling the signal will be decimated using scipy.signal.decimate
  """

  if k < 1:
    raise NotImplementedError ('downsampling is not implemented')

  if k == 1:
    return x # nothing to do

  return upsample (x, k)

def upsample (x, k):
  """
  Upsample the signal to the given ratio using a sinc kernel

  input:

    x   a vector or matrix with a signal in each row
    k   ratio to resample to

    returns

    y   the up or downsampled signal

    when downsampling the signal will be decimated using scipy.signal.decimate
  """

  assert k >= 1, 'k must be equal or greater than 1'

  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = n * k

  xt = np.linspace (1, n, n)
  xp = np.linspace (1, n, nn)

  return interp (xp, xt, x)

def upsample3 (x, k, workers = None):
  """
  Like upsample, but uses the multi-threaded interp3
  """

  assert k >= 1, 'k must be equal or greater than 1'

  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = n * k

  xt = np.linspace (1, n, n)
  xp = np.linspace (1, n, nn)

  return interp3 (xp, xt, x, workers)


def interp (xp, xt, x):
  """
  Interpolate the signal to the new points using a sinc kernel

  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on

  output:
  y     the interpolated signal at points xp
  """

  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = len(xp)

  y = np.zeros((m, nn))

  for (pi, p) in enumerate (xp):
    si = np.tile(np.sinc (xt - p), (m, 1))
    y[:, pi] = np.sum(si * x)

  return y.squeeze ()

default_workers = 6
def interp3 (xp, xt, x, workers = default_workers):
  """
  Interpolate the signal to the new points using a sinc kernel

  Like interp, but splits the signal into domains and calculates them
  separately using multiple threads.

  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on
  workers  number of threaded workers to use (default: 16)

  output:
  y     the interpolated signal at points xp
  """

  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = len(xp)

  y = np.zeros((m, nn))

  # from upsample
  if workers is None: workers = default_workers

  xxp = np.array_split (xp, workers)

  from concurrent.futures import ThreadPoolExecutor
  import concurrent.futures

  def approx (_xp, strt):
    for (pi, p) in enumerate (_xp):
      si = np.tile (np.sinc (xt - p), (m, 1))
      y[:, strt + pi] = np.sum (si * x)

  jobs = []
  with ThreadPoolExecutor (max_workers = workers) as executor:
    strt = 0
    for w in np.arange (0, workers):
      f = executor.submit (approx, xxp[w], strt)
      strt = strt + len (xxp[w])
      jobs.append (f)


  concurrent.futures.wait (jobs)

  return y.squeeze ()

def upsample2 (x, k):
  """
  Upsample the signal to the new points using a sinc kernel. The
  interpolation is done using a matrix multiplication.

  Requires a lot of memory, but is fast.

  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on

  output:
  y     the interpolated signal at points xp
  """

  mn = x.shape

  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = n * k

  [T, Ts]  = np.mgrid[1:n:nn*1j, 1:n:n*1j]
  TT = Ts - T
  del T, Ts

  y = np.sinc(TT).dot (x.reshape(n, 1))

  return y.squeeze()


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  F     = 10.
  Fs    = 100. * F
  dt    = 1. / Fs

  t = np.arange (0,  2 * np.pi, dt)
  s = np.sin (F * t)


  mFs   = 3 * F
  mt    = np.arange (min(t), max(t), 1. / mFs)
  ms    = np.sin (F * mt)

  k = 16
  us = upsample3 (ms, k)
  ut = np.linspace (min(mt), max(mt), len(us))

  plt.figure ()
  plt.plot (t, s, 'b', label = 'high Fs')
  plt.plot (mt, ms, 'r', label = 'low Fs')
  plt.plot (ut, us, 'g', label = 'resampled')


  # diff
  uus = np.interp (t, ut, us)
  dd  = uus - s

  plt.plot (t, dd, 'k', label = 'diff')

  plt.legend ()

  plt.show ()


