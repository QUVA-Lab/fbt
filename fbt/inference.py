"""Inference methods for contact-graphs."""
from fbt import constants, logger, util
import numba
import numpy as np
import time
from typing import Any, Optional, Tuple


@numba.njit
def softmax(x):
  y = x - np.max(x)
  return np.exp(y)/np.sum(np.exp(y))


@numba.njit(parallel=True)
def fn_step_wrapped(
    user_interval: Tuple[int, int],
    seq_array_hot: np.ndarray,
    log_c_z_u: np.ndarray,
    log_A_start: np.ndarray,
    p_infected_matrix: np.ndarray,
    num_time_steps: int,
    probab0: float,
    probab1: float,
    past_contacts_array: np.ndarray,
    start_belief: np.ndarray,
    quantization: int = -1,):
  """Wraps one step of Factorised Neighbors over a subset of users.

  Args:
    user_interval: tuple of (user_start, user_end)
    seq_array_hot: array in [num_time_steps, 4, num_sequences]
    log_c_z_u: array in [num_users_int, num_sequences], C-terms according to
      CRISP paper
    log_A_start: array in [num_sequences], A-terms according to CRISP paper
    p_infected_matrix: array in [num_users, num_time_steps]
    num_time_steps: number of time steps
    probab0: probability of transitioning S->E
    probab1: probability of transmission given contact
    past_contacts_array: iterator with elements (timestep, user_u, features)
    start_belief: matrix in [num_users_int, 4], i-th row is assumed to be the
      start_belief of user user_slice[i]
    quantization: number of quantization levels
  """
  with numba.objmode(t0='f8'):
    t0 = time.time()

  # Apply quantization
  if quantization > 0:
    p_infected_matrix = util.quantize_floor(
      p_infected_matrix, num_levels=quantization)

  p_infected_matrix = p_infected_matrix.astype(np.float32)

  interval_num_users = user_interval[1] - user_interval[0]

  post_exps = np.zeros((interval_num_users, num_time_steps, 4),
    dtype=np.float32)
  start_belief = start_belief.astype(np.float32)
  num_days_s = np.sum(seq_array_hot[:, 0], axis=0).astype(np.int32)

  assert np.all(np.sum(seq_array_hot, axis=1) == 1), (
    "seq_array_hot is expected as one-hot array")

  seq_array_hot = seq_array_hot.astype(np.single)
  num_sequences = seq_array_hot.shape[2]

  # Array in [4, num_sequences]
  state_start_hot = seq_array_hot[0]
  # Array in [num_users, num_sequences]
  start_belief_all = np.log(start_belief.dot(state_start_hot) + 1E-12)

  for i in numba.prange(interval_num_users):  # pylint: disable=not-an-iterable

    d_term, d_no_term = util.precompute_d_penalty_terms_fn2(
      p_infected_matrix,
      p0=probab0,
      p1=probab1,
      past_contacts=past_contacts_array[i],
      num_time_steps=num_time_steps)
    d_noterm_cumsum = np.cumsum(d_no_term)

    d_penalties = (
      np.take(d_noterm_cumsum, np.maximum(num_days_s-1, 0))
      + np.take(d_term, num_days_s))

    # Calculate log_joint
    # Numba only does matmul with 2D-arrays, so do reshaping below
    log_joint = log_c_z_u[i] + log_A_start + d_penalties + start_belief_all[i]

    joint_distr = softmax(log_joint).astype(np.single)
    post_exps[i] = np.reshape(np.dot(
      seq_array_hot.reshape(num_time_steps*4, num_sequences), joint_distr),
      (num_time_steps, 4))

  with numba.objmode(t1='f8'):
    t1 = time.time()

  return post_exps, t0, t1


def fact_neigh(
    num_users: int,
    num_time_steps: int,
    observations_all: constants.ObservationList,
    contacts_all: constants.ContactList,
    probab_0: float,
    probab_1: float,
    g_param: float,
    h_param: float,
    start_belief: Optional[np.ndarray] = None,
    alpha: float = 0.001,
    beta: float = 0.01,
    quantization: int = -1,
    users_stale: Optional[np.ndarray] = None,
    num_updates: int = 1000,
    verbose: bool = False,
    diagnostic: Optional[Any] = None) -> np.ndarray:
  """Inferes latent states using Factorised Neighbor method.

  Uses Factorised Neighbor approach from
  'The dlr hierarchy of approximate inference, Rosen-Zvi, Jordan, Yuille, 2012'

  Args:
    num_users: Number of users to infer latent states
    num_time_steps: Number of time steps to infer latent states
    observations_all: List of all observations
    contacts_all: List of all contacts
    probab_0: Probability to be infected spontaneously
    probab_1: Probability of transmission given contact
    g_param: float, dynamics parameter, p(E->I|E)
    h_param: float, dynamics parameter, p(I->R|I)
    start_belief: array in [num_users, 4], which are the beliefs for the start
      state
    alpha: False positive rate of observations, (1 minus specificity)
    beta: False negative rate of observations, (1 minus sensitivity)
    quantization: number of levels for quantization. Negative number indicates
      no use of quantization.
    num_updates: Number of rounds to update using Factorised Neighbor algorithm
    verbose: set to true to get more verbose output

  Returns:
    array in [num_users, num_timesteps, 4] being probability of over
    health states {S, E, I, R} for each user at each time step
  """
  del diagnostic
  t_start_preamble = time.time()

  # Slice out stale_users
  users_stale_binary = util.get_stale_users_binary(
    users_stale, num_users)

  seq_array = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))
  seq_array_hot = np.transpose(util.state_seq_to_hot_time_seq(
    seq_array, time_total=num_time_steps), [1, 2, 0]).astype(np.int8)

  # If 'start_belief' is provided, the prior will be applied per user, later
  if start_belief is None:
    prior = [1-probab_0, probab_0, 0., 0.]
  else:
    prior = [.25, .25, .25, .25]

  log_A_start = util.enumerate_log_prior_values(
    prior, [1-probab_0, 1-g_param, 1-h_param],
    seq_array, num_time_steps)

  obs_array = util.make_inf_obs_array(int(num_time_steps), alpha, beta)
  # Precompute log(C) terms, relating to observations
  log_c_z_u = util.calc_c_z_u(
    (int(0), int(num_users)),
    obs_array,
    observations_all)

  q_marginal_infected = np.zeros((num_users, num_time_steps), dtype=np.single)
  post_exp = np.zeros((num_users, num_time_steps, 4), dtype=np.single)

  t_preamble1 = time.time() - t_start_preamble
  t_start_preamble = time.time()

  num_max_msg = int(100*num_time_steps)
  past_contacts, max_num_contacts = util.get_past_contacts_static(
    (int(0), int(num_users)), contacts_all, num_msg=num_max_msg)

  if max_num_contacts >= num_max_msg:
    logger.warning(
      f"Max number of contacts {max_num_contacts} >= {num_max_msg}")

  start_belief_matrix = np.ones((num_users, 4), dtype=np.single)
  if start_belief is not None:
    assert len(start_belief) == num_users
    start_belief_matrix = start_belief

  t_preamble2 = time.time() - t_start_preamble
  logger.info(
    f"Time spent on preamble1/preamble2 {t_preamble1:.1f}/{t_preamble2:.1f}")

  for num_update in range(num_updates):
    if verbose:
      logger.info(f"Num update {num_update}")

    if users_stale is not None:
      q_marginal_infected *= users_stale_binary

    post_exp, tstart, t_end = fn_step_wrapped(
      (int(0), int(num_users)),
      seq_array_hot,
      log_c_z_u,
      log_A_start,
      q_marginal_infected,
      num_time_steps,
      probab_0,
      probab_1,
      past_contacts_array=past_contacts,
      start_belief=start_belief_matrix,
      quantization=quantization)

    if verbose:
      logger.info(f"Time for fn_step: {t_end - tstart:.1f} seconds")

    q_marginal_infected = post_exp[:, :, 2]

  return post_exp.astype(np.float32)
