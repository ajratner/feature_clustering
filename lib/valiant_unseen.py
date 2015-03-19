# Implementation of Valiant et. al. from MATLAB code
import numpy as np
from scipy.stats import poisson
import cvxpy as cvx

# given a fingerprint (numpy) vector: 
#   F = [F_1,...,F_m], where F_i is the number of elements seen i times in a sample
# return the histogram and histogram support set for the estimated distribution D:
#   h_D(x) = |{\alpha : D(\alpha) = x}|
def unseen(f):
  N_sample = float(sum(np.arange(1,len(f)+1)*f))

  # PARAMETERS
  gridFactor = 1.1
  alpha = 0.5
  xLPmin = 1/(N_sample*max(10,N_sample))
  maxLPIters = 1000

  # Split f into a 'dense' portion (LP used) & a 'sparse' portion (empirical histogram used)
  x = np.array([])
  histx = np.array([])
  fLP = np.zeros(len(f))
  for i,f_i in enumerate(f):
    if f_i > 0:
      window = [int(max(1, i-np.ceil(np.sqrt(i)))), int(min(i+np.ceil(np.sqrt(i)), len(f)))]
      if sum([f[j] for j in range(window[0], window[1])]) < 2*np.sqrt(i):
        x = np.append(x, i/N_sample)
        histx = np.append(histx, f_i)
        fLP[i] = 0;
      else:
        fLP[i] = f_i

  # if no LP portion return
  if len(filter(lambda i : i > 0, fLP)) == 0:
    return x[2:], histx[2:]

  # set up LP1
  LPmass = 1 - np.dot(x, histx) # amount of probability mass in the LP region
  fmax = max([i for i,flp in enumerate(fLP) if flp > 0]) + 1
  fLP = np.concatenate((fLP[:fmax], np.zeros(int(np.ceil(np.sqrt(fmax))))))
  szLPf = len(fLP)
  xLPmax = fmax/N_sample
  xLP = xLPmin*np.array([gridFactor**p for p in range(int(np.ceil(np.log(xLPmax/xLPmin)/np.log(gridFactor)))+1)])
  szLPx = len(xLP)

  # we minimize C.T*x with constraints Ax <= b, Aeq = beq, lb <= x <= ub
  C = np.zeros(int(szLPx + 2*szLPf))
  for i in range(len(fLP)):
    C[szLPx+2*i] = 1.0/np.sqrt(fLP[i]+1)
  for i in range(len(fLP)):
    C[szLPx+1+2*i] = 1.0/np.sqrt(fLP[i]+1)
  A = np.zeros((2*szLPf,szLPx+2*szLPf))
  b = np.zeros((2*szLPf,1))
  for i in range(szLPf):
    A[2*i,:szLPx] = poisson.pmf(i+1,N_sample*xLP)
    A[2*i+1,:szLPx] = (-1)*A[2*i,:szLPx]
    A[2*i,szLPx+2*i] = -1
    A[2*i+1,szLPx+2*i+1] = -1
    b[2*i] = fLP[i]
    b[2*i+1] = -fLP[i]
  Aeq = np.zeros((1,szLPx+2*szLPf))
  Aeq[0,:szLPx] = xLP
  beq = LPmass
  for i in range(szLPx):
    A[:,i] = A[:,i]/xLP[i]
    Aeq[0,i] = Aeq[0,i]/xLP[i]

  # solve LP1
  xvar = cvx.Variable(C.shape[0])
  constraints = [A*xvar <= b,
                 Aeq*xvar == beq,
                 0 <= xvar]
  opt_funct = cvx.sum_entries(cvx.mul_elemwise(C, xvar))
  obj = cvx.Minimize(opt_funct)
  prob = cvx.Problem(obj, constraints)
  prob.solve(solver=cvx.ECOS, max_iters=maxLPIters)
  if prob.value is None or xvar.value is None:
    print "LP1 solution not found."
    return None, None

  # set up LP2
  C2 = 0*C
  C2[:szLPx] = 1
  A2 = np.vstack((A, C.T))
  b2 = np.vstack((b, prob.value+alpha))
  for i in range(szLPx):
    C2[i] = C2[i]/xLP[i]

  # solve LP2
  xvar2 = cvx.Variable(C2.shape[0])
  constraints2 = [A2*xvar2 <= b2,
                 Aeq*xvar2 == beq,
                 0 <= xvar2]
  opt_funct2 = cvx.sum_entries(cvx.mul_elemwise(C2, xvar2))
  obj2 = cvx.Minimize(opt_funct2)
  prob2 = cvx.Problem(obj2, constraints2)
  prob2.solve(solver=cvx.ECOS, max_iters=maxLPIters)
  if prob2.value is None or xvar2.value is None:
    print "LP2 solution not found, returning with LP1 value..."

  # append LP solution to empirical portion of histogram
  xv2 = xvar2.value if xvar2.value is not None else xvar.value
  xv2[:szLPx] = xv2[:szLPx,:] / np.reshape(xLP, (len(xLP), 1))
  x = np.concatenate((x,xLP))
  histx = np.concatenate((histx, np.ravel(xv2.T)))
  idx = np.argsort(x)
  x = x[idx]
  histx = histx[idx]
  idx = [i for i,hx in enumerate(histx) if hx > 0]
  histx = histx[idx]
  x = x[idx]
  return histx, x


def prob_unseen(f):
  histx, x = unseen(f)
  if histx is None:
    return None
  else:
    return 1.0 - (sum(f)/float(sum(histx)))
