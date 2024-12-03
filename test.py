import numpy as np
import pandas as pd
import statistics
import sklearn.datasets

import sys
sys.path.append('../')

import data_loader
from dp_regression import tukey
import time

from sklearn.metrics import r2_score, root_mean_squared_error

def adassp(features, labels, epsilon, delta, rho=0.05):
  """Returns model computed using AdaSSP DP linear regression.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Computed model satisfies (epsilon, delta)-DP.
    delta: Computed model satisfies (epsilon, delta)-DP.
    rho: Failure probability. The default of 0.05 is the one used in
      https://arxiv.org/pdf/1803.02596.pdf.

  Returns:
    Vector of regression coefficients. AdaSSP is described in Algorithm 2 of
    https://arxiv.org/pdf/1803.02596.pdf.
  """
  _, d = features.shape
  # these bounds are data-dependent and not dp
  bound_x = np.amax(np.linalg.norm(features, axis=1))
  # bound_y = np.amax(np.abs(labels))
  bound_y = 0
  for i in range(len(labels)):
     bound_y = max(bound_y, np.abs(labels[i]))
  lambda_min = max(0,
                   np.amin(np.linalg.eigvals(np.matmul(features.T, features))))
  z = np.random.normal(size=1)
  sensitivity = np.sqrt(np.log(6 / delta)) / (epsilon / 3)
  private_lambda = max(
      0, lambda_min + sensitivity * (bound_x**2) * z -
      (bound_x**2) * np.log(6 / delta) / (epsilon / 3))
  final_lambda = max(
      0,
      np.sqrt(d * np.log(6 / delta) * np.log(2 * (d**2) / rho)) * (bound_x**2) /
      (epsilon / 3) - private_lambda)
  # generate symmetric noise_matrix where each upper entry is iid N(0,1)
  noise_matrix = np.random.normal(size=(d, d))
  noise_matrix = np.triu(noise_matrix)
  noise_matrix = noise_matrix + noise_matrix.T - np.diag(np.diag(noise_matrix))
  priv_xx = np.matmul(features.T,
                      features) + sensitivity * (bound_x**2) * noise_matrix
  priv_xy = np.matmul(features.T, labels).flatten() + sensitivity * bound_x * bound_y * np.random.normal(size=d)
  model_adassp = np.matmul(
      np.linalg.pinv(priv_xx + final_lambda * np.eye(d)), priv_xy)
  return model_adassp

def r_squared_from_models(models, features, labels):
  predictions = np.matmul(models, features.T)
  r2=[]
  rmse = []
  for i in range(predictions.shape[0]):
     cur_pred = predictions[i]
     r2.append(r2_score(labels, cur_pred))
     rmse.append(root_mean_squared_error(labels, cur_pred))
  #r2_vals = r_squared(predictions, labels)
  #return np.quantile(r2_vals, 0.25, axis=0), np.quantile(r2_vals, 0.5, axis=0), np.quantile(r2_vals, 0.75, axis=0)
  return r2, rmse

def run_dp_tukey(features, labels, epsilon, delta, m, num_trials):
  _, d = features.shape
  final_models = np.zeros((num_trials, d))
  for trial in range(num_trials):
    models = tukey.multiple_regressions(features, labels, m)
    final_models[trial, :] = tukey.dp_tukey(models, epsilon, delta)
  return r_squared_from_models(final_models, features, labels)

def run_adassp(features, labels, epsilon, delta, num_trials):
  """Returns 0.25, 0.5, and 0.75 quantiles from num_trials AdaSSP models.

  Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Each DP model satisfies (epsilon, delta)-DP.
    delta: Each DP model satisfies (epsilon, delta)-DP.
    num_trials: Number of trials to run.
  """
  models = np.zeros((num_trials, len(features[0])))
  for trial in range(num_trials):
    models[trial, :] = adassp(features, labels, epsilon, delta)
  return r_squared_from_models(models, features, labels)


# generate synthetic data
#synthetic_x, synthetic_y = sklearn.datasets.make_regression(
#    22000, n_features=10, n_informative=10, noise=10)
#synthetic_x = np.column_stack(
#    (synthetic_x, np.ones(synthetic_x.shape[0]))).astype(float)

synthetic_x, synthetic_y, _ = data_loader.gen_cal_housing(normalization = True)

# set remaining parameters
epsilon = 0.2
delta = 1e-5
m_range = np.linspace(2000, 3000, 5)
num_trials = 10


# run r2 experiments and store r2 quantiles and times


features = synthetic_x
labels = synthetic_y

(n, d) = features.shape
num_m = len(m_range)

times = np.zeros(num_m)

begin = time.time()
r2, rmse = run_adassp(features, labels, epsilon, delta,
                                          num_trials)
end = time.time()
print("finished adassp")

print("adassp results: r2_stats = ", [np.mean(r2), np.std(r2)], ", rmse_stats = ", [np.mean(rmse), np.std(rmse)])

r2_list = []
rmse_list = []
times = np.zeros(num_m)

for m_idx in range(num_m):
    m = int(m_range[m_idx])
    batch_size = int(n / m)
    if batch_size < d:
        raise RuntimeError(
              str(m) + " models requires " + str(m * d) +
              " points, but given features only has " + str(n) + " points.")
    print("finished tukey, m = " + str(m))
    begin = time.time()
    r2, rmse = run_dp_tukey(features, labels, epsilon, delta, m,
                                  num_trials)
    r2_list.append(r2)
    rmse_list.append(rmse)
    end = time.time()
    times[m_idx] = (end - begin) / num_trials
    print("r2_stats = ", [np.mean(r2), np.std(r2)], ", rmse_stats = ", [np.mean(rmse), np.std(rmse)])
    
"""
r2_stats = np.zeros(shape=(num_m, 2))
rmse_stats = np.zeros(shape=(num_m, 2))

for m_idx in range(num_m):
  r2_stats[m_idx] = [np.mean(r2_list[m_idx]), np.std(r2_list[m_idx])]
  rmse_stats[m_idx] = [np.mean(rmse_list[m_idx]), np.std(rmse_list[m_idx])]
"""

# print(r2_list)
# print(rmse_list)
# print(times)
