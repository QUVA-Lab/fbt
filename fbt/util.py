"""Utility functions for inference in CRISP-like models."""

import contextlib
import functools
import itertools
import time  # pylint: disable=unused-import
import math
from fbt import constants, logger
import numba
import numpy as np
import os
import socket
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union


class InfectiousContactCount:
  """Counter for infectious contacts."""

  def __init__(self,
               contacts: constants.ContactList,
               samples: Optional[Mapping[int, Union[np.ndarray, List[int]]]],
               num_users: int,
               num_time_steps: int):
    self._counts = np.zeros((num_users, num_time_steps + 1), dtype=np.int32)
    self._num_time_steps = num_time_steps

    # TODO: WARNING: This constructor assumes that the contacts don't change!
    self.past_contacts = {
      user: [[] for _ in range(num_time_steps)] for user in range(num_users)}
    self.future_contacts = {
      user: [[] for _ in range(num_time_steps)] for user in range(num_users)}

    for contact in contacts:
      user_u = contact[0]
      user_v = contact[1]
      timestep = contact[2]
      if timestep < num_time_steps:
        self.past_contacts[user_v][timestep].append(
          (timestep, user_u, contact[3]))
        self.future_contacts[user_u][timestep].append(
          (timestep, user_v, contact[3]))

      if samples:
        trace_u = samples[user_u]
        if state_at_time_cache(*trace_u, contact[2]) == 2:
          self._counts[user_v][contact[2]+1] += 1

  def get_past_contacts_slice(
      self, user_slice: Union[List[int], np.ndarray]) -> np.ndarray:
    """Outpets past contacts as a NumPy array, for easy pickling.

    Defaults to -1 when no past contacts exist (and to fill sparse array)
    """
    past_contacts = []
    max_messages = -1  # Output ndarray will be size of longest list

    for user in user_slice:
      pc_it = itertools.chain.from_iterable(self.past_contacts[user])
      pc_array = np.array(
        list(map(lambda x: [x[0], x[1], x[2]], pc_it)), dtype=np.int32)
      past_contacts.append(pc_array)

      # Update longest amount of messages
      max_messages = max((max_messages, len(pc_array)))

    # Default to -1 for undefined past contacts
    pc_tensor = -1 * np.ones(
      (len(user_slice), max_messages+1, 3), dtype=np.int32)
    for i, user in enumerate(user_slice):
      num_contacts = len(past_contacts[i])
      if num_contacts > 0:
        pc_tensor[i][:num_contacts] = past_contacts[i]

    return pc_tensor

  def num_inf_contacts(self, user: int, time_step: int):
    """Returns the number of infectious contacts at a given time step."""
    return self._counts[user, time_step]

  def update_infect_count(
      self, user: int, trace: Union[List[int], np.ndarray],
      remove: bool = False):
    """Updates the infectious contact count for a given user.

    Infectious counts are updated for the neighbors of the user.
    """
    t0, de, di = trace
    for timestep in range(t0+de, t0+de+di):
      for (_, user_v, feature) in self.future_contacts[user][timestep]:
        assert feature == 1, "Only implemented for feature_val==1"
        add = -1 if remove else 1
        self._counts[user_v][timestep+1] += add

  def get_future_contacts(self, user: int):
    """Yields the future contacts of a user."""
    yield from itertools.chain.from_iterable(self.future_contacts[user])

  def get_past_contacts(self, user: int):
    """Yields the past contacts of a user."""
    yield from itertools.chain.from_iterable(self.past_contacts[user])

  def get_past_contacts_at_time(self, user: int, timestep: int):
    """Yields the past contacts of a user at a given time step."""
    yield from self.past_contacts[user][timestep]


# @numba.njit
def get_past_contacts_fast(
    user_interval: Tuple[int, int],
    contacts: np.ndarray) -> Tuple[np.ndarray, int]:
  """Returns past contacts as a NumPy array, for easy pickling."""
  num_users_int = user_interval[1] - user_interval[0]

  if len(contacts) == 0:
    return -1 * np.ones((num_users_int, 1, 3), dtype=np.int32), 0
  if contacts.shape[1] == 0:
    return -1 * np.ones((num_users_int, 1, 3), dtype=np.int32), 0

  contacts_past = [[(-1, -1, -1)] for _ in range(num_users_int)]

  # First find all contacts that are in the interval
  for contact in contacts:
    user_v = contact[1]
    if user_interval[0] <= user_v < user_interval[1]:
      datum = (contact[2], contact[0], contact[3])
      contacts_past[user_v - user_interval[0]].append(datum)

  # Then construct Numpy array
  max_messages = max(map(len, contacts_past)) - 1  # Subtract 1 to make clear!
  pc_tensor = -1 * np.ones((num_users_int, max_messages + 1, 3), dtype=np.int32)
  for i in range(num_users_int):
    num_contacts = len(contacts_past[i])
    if num_contacts > 1:
      pc_array = np.array(contacts_past[i][1:], dtype=np.int32)
      pc_tensor[i][:num_contacts-1] = pc_array

  return pc_tensor, max_messages


@numba.njit
def get_past_contacts_static(
    user_interval: Tuple[int, int],
    contacts: np.ndarray,
    num_msg: int) -> Tuple[np.ndarray, int]:
  """Returns past contacts as a NumPy array, for easy pickling."""
  num_users_int = user_interval[1] - user_interval[0]

  if len(contacts) == 0:
    return (-1 * np.ones((num_users_int, 1, 3))).astype(np.int32), 0
  if contacts.shape[1] == 0:
    return (-1 * np.ones((num_users_int, 1, 3))).astype(np.int32), 0

  contacts_past = -1 * np.ones((num_users_int, num_msg, 3), dtype=np.int32)

  contacts_counts = np.zeros(num_users_int, dtype=np.int32)

  # First find all contacts that are in the interval
  for contact in contacts:
    user_v = contact[1]
    if user_interval[0] <= user_v < user_interval[1]:
      contact_rel = user_v - user_interval[0]
      contact_count = contacts_counts[contact_rel] % (num_msg - 1)
      contacts_past[contact_rel, contact_count] = np.array(
        (contact[2], contact[0], contact[3]), dtype=np.int32)

      contacts_counts[contact_rel] += 1

  return contacts_past.astype(np.int32), int(np.max(contacts_counts))


def state_at_time(days_array, timestamp):
  """Calculates the SEIR state at timestamp given the Markov state.

  Note that this function is slower than 'state_at_time_cache' when evaluating
  only one data point.
  """
  if isinstance(days_array, list):
    days_array = np.array(days_array, ndmin=2)
  elif len(days_array.shape) == 1:
    days_array = np.expand_dims(days_array, axis=0)

  days_cumsum = np.cumsum(days_array, axis=1)
  days_binary = (days_cumsum <= timestamp).astype(np.int)

  # Append vector of 1's such that argmax defaults to 3
  days_binary = np.concatenate(
    (days_binary, np.zeros((len(days_binary), 1))), axis=1)
  state = np.argmin(days_binary, axis=1)
  return state


# @functools.lru_cache()  # Using cache gives no observable speed up for now
def state_at_time_cache(t0: int, de: int, di: int, t: int) -> int:
  if t < t0:
    return 0
  if t < t0+de:
    return 1
  if t < t0+de+di:
    return 2
  return 3


def gather_infected_precontacts(
    num_time_steps: int,
    samples_current: Mapping[int, Union[List[int], np.ndarray]],
    past_contacts: Iterable[Tuple[int, int, List[int]]]):
  """Gathers infected precontacts.

  For all past contacts, check if the contact was infected according to samples.
  """
  num_infected_preparents = np.zeros((num_time_steps))

  for (t_contact, user_u, feature) in past_contacts:
    assert feature == 1, (
      "Code only implemented for singleton feature at 1")
    trace_u = samples_current[user_u]
    state_u = state_at_time_cache(*trace_u, t_contact)
    if state_u == 2:
      num_infected_preparents[t_contact] += feature

  return num_infected_preparents


@numba.njit
def calc_c_z_u(
    user_interval: Tuple[int, int],
    obs_array: np.ndarray,
    observations: np.ndarray) -> np.ndarray:
  """Precompute the Cz terms.

  Args:
    user_interval: Tuple of (start, end) user indices.
    obs_array: Array in [num_time_steps, num_sequences, 2] of observations.
      obs_array[t, :, i] is about the log-likelihood of the observation being
      i=0 or i=1, at time step t.
    observations: Array in [num_observations, 3] of observations.

  Notation follows the original CRISP paper.
  """
  interval_num_users = user_interval[1] - user_interval[0]
  log_prob_obs = np.zeros((interval_num_users, obs_array.shape[1]))

  if observations.shape[1] > 1:
    # Only run if there are observations
    for obs in observations:
      user_u = obs[0]

      if user_interval[0] <= user_u < user_interval[1]:
        log_prob_obs[user_u - user_interval[0]] += obs_array[obs[1], :, obs[2]]

  return log_prob_obs


def calc_log_a_start(
    seq_array: np.ndarray,
    probab_0: float,
    g: float,
    h: float) -> np.ndarray:
  """Calculate the basic A terms.

  This assumes no contacts happen. Thus the A terms are simple Geometric
  distributions. When contacts do occur, subsequent code would additional log
  terms.
  """
  if isinstance(seq_array, list):
    seq_array = np.stack(seq_array, axis=0)

  num_sequences = seq_array.shape[0]
  log_A_start = np.zeros((num_sequences))

  time_total = np.max(np.sum(seq_array, axis=1))

  # Due to time in S
  # Equation 17
  term_t = (seq_array[:, 0] >= time_total).astype(np.float32)
  log_A_start += (seq_array[:, 0]-1) * np.log(1-probab_0)

  # Due to time in E
  term_e = ((seq_array[:, 0] + seq_array[:, 1]) >= time_total).astype(np.int32)
  log_A_start += (1-term_t) * (
    (seq_array[:, 1]-1)*np.log(1-g) + (1-term_e)*np.log(g)
  )

  # Due to time in I
  term_i = (seq_array[:, 0] + seq_array[:, 1] + seq_array[:, 2]) >= time_total
  log_A_start += (1-term_e) * (
    (seq_array[:, 2]-1)*np.log(1-h) + (1-term_i.astype(np.int32))*np.log(h))
  return log_A_start


def iter_state(ts, te, ti, tt):
  yield from itertools.repeat(0, ts)
  yield from itertools.repeat(1, te)
  yield from itertools.repeat(2, ti)
  yield from itertools.repeat(3, tt-ts-te-ti)


def state_seq_to_time_seq(
    state_seqs: Union[np.ndarray, List[List[int]]],
    time_total: int) -> np.ndarray:
  """Unfolds trace tuples to full traces of SEIR.

  Args:
    state_seqs: np.ndarray in [num_sequences, 3] with columns for t_S, t_E, t_I
    time_total: total amount of days. In other words, each row sum is
      expected to be less than time_total, and including t_R should equal
      time_total.

  Returns:
    np.ndarray of [num_sequences, time_total], with values in {0,1,2,3}
  """
  iter_state_partial = functools.partial(iter_state, tt=time_total)
  iter_time_seq = map(list, itertools.starmap(iter_state_partial, state_seqs))
  return np.array(list(iter_time_seq))


def state_seq_to_hot_time_seq(
    state_seqs: Union[np.ndarray, List[List[int]]],
    time_total: int) -> np.ndarray:
  """Unfolds trace tuples to one-hot traces of SEIR.

  Args:
    state_seqs: np.ndarray in [num_sequences, 3] with columns for t_S, t_E, t_I
    time_total: total amount of days. In other words, each row sum is
      expected to be less than time_total, and including t_R should equal
      time_total.

  Returns:
    np.ndarray of [num_sequences, time_total, 4], with values in {0,1}
  """
  iter_state_partial = functools.partial(iter_state, tt=time_total)
  iter_time_seq = map(list, itertools.starmap(iter_state_partial, state_seqs))

  states = np.zeros((len(state_seqs), time_total, 4))
  for i, time_seq in enumerate(iter_time_seq):
    states[i] = np.take(np.eye(4), np.array(time_seq), axis=0)
  return states


def iter_sequences(time_total: int, start_se=True):
  """Iterate possible sequences.

  Assumes that first time step can be either S or E.
  """
  for t0 in range(time_total+1):
    if t0 == time_total:
      yield (t0, 0, 0)
    else:
      e_start = 1 if (t0 > 0 or start_se) else 0
      for de in range(e_start, time_total-t0+1):
        if t0+de == time_total:
          yield (t0, de, 0)
        else:
          i_start = 1 if (t0 > 0 or de > 0 or start_se) else 0
          for di in range(i_start, time_total-t0-de+1):
            if t0+de+di == time_total:
              yield (t0, de, di)
            else:
              yield (t0, de, di)


def generate_sequence_days(time_total: int):
  """Iterate possible sequences.

  Assumes that first time step must be S.
  """
  # t0 ranges in {T,T-1,...,1}
  for t0 in range(time_total, 0, -1):
    # de ranges in {T-t0,T-t0-1,...,1}
    # de can only be 0 when time_total was already spent
    de_start = min((time_total-t0, 1))
    non_t0 = time_total - t0
    for de in range(de_start, non_t0+1):
      # di ranges in {T-t0-de,T-t0-de-1,...,1}
      # di can only be 0 when time_total was already spent
      di_start = min((time_total-t0-de, 1))
      non_t0_de = time_total - t0 - de
      for di in range(di_start, non_t0_de+1):
        yield (t0, de, di)


@functools.lru_cache(maxsize=1)
def make_inf_obs_array(
    num_time_steps: int, alpha: float, beta: float) -> np.ndarray:
  """Makes an array with observation log-terms per day.

  Obs_array is of shape [num_time_steps, num_sequences, 2], where the last
  dimension is about the log-likelihood of the observation being 0 or 1.
  """
  pot_seqs = np.stack(list(
    iter_sequences(time_total=num_time_steps, start_se=False)))
  time_seqs = state_seq_to_time_seq(pot_seqs, num_time_steps)

  out_array = np.zeros((num_time_steps, len(pot_seqs), 2))
  for t in range(num_time_steps):
    out_array[t, :, 0] = np.log(
      np.where(time_seqs[:, t] == 2, alpha, 1-beta))
    out_array[t, :, 1] = np.log(
      np.where(time_seqs[:, t] == 2, 1-alpha, beta))
  return out_array


def enumerate_log_prior_values(
    params_start: Union[np.ndarray, List[float]],
    params: Union[np.ndarray, List[float]],
    sequences: np.ndarray,
    time_total: int) -> np.ndarray:
  """Enumerate values of log prior."""
  np.testing.assert_almost_equal(np.sum(params_start), 1.)

  feature_weight_0, feature_weight_1, feature_weight_2 = (
    params[0], params[1], params[2])

  start_s = reach_s = sequences[:, 0] > 0
  start_e = (1-start_s) * (sequences[:, 1] > 0)
  start_i = (1-start_s) * (1-start_e) * (sequences[:, 2] > 0)
  start_r = (1-start_s) * (1-start_e) * (1-start_i)

  reach_e = (sequences[:, 1] > 0)
  reach_i = (sequences[:, 2] > 0)
  reach_r = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) < time_total

  log_q_z = np.zeros((len(sequences)))

  # Terms due to start state
  log_q_z += start_s * np.log(params_start[0] + 1E-12)
  log_q_z += start_e * np.log(params_start[1] + 1E-12)
  log_q_z += start_i * np.log(params_start[2] + 1E-12)
  log_q_z += start_r * np.log(params_start[3] + 1E-12)

  # Terms due to days spent in S
  log_q_z += np.maximum(
    sequences[:, 0]-1, 0.) * np.log(feature_weight_0)
  log_q_z += reach_s * reach_e * np.log(
    1-feature_weight_0)  # Only when transit to E is made

  # Terms due to days spent in E
  log_q_z += np.maximum(sequences[:, 1] - 1, 0.) * np.log(feature_weight_1)
  log_q_z += reach_e * reach_i * np.log(
    1-feature_weight_1)  # Only when transit to I is made

  # Terms due to days spent in I
  log_q_z += np.maximum(sequences[:, 2] - 1, 0.) * np.log(feature_weight_2)
  log_q_z += reach_i * reach_r * np.log(
    1-feature_weight_2)  # Only when transit to R is made

  return log_q_z


def enumerate_start_belief(
    seq_array: np.ndarray, start_belief: np.ndarray) -> np.ndarray:
  """Calculates the start_belief for all enumerated sequences."""
  assert seq_array.shape[1] == 3
  assert start_belief.shape == (4,)

  start_s = seq_array[:, 0] > 0
  start_e = (1.-start_s) * (seq_array[:, 1] > 0)
  start_i = (1.-start_s) * (1.-start_e) * (seq_array[:, 2] > 0)
  start_r = np.sum(seq_array, axis=1) == 0

  return (
    start_belief[0] * start_s
    + start_belief[1] * start_e
    + start_belief[2] * start_i
    + start_belief[3] * start_r)


def enumerate_log_q_values(
    params: np.ndarray,
    sequences: np.ndarray) -> np.ndarray:
  """Enumerate values of log_q for variational parameters."""
  a0, feature_weight_0, feature_weight_1, feature_weight_2 = (
    params[0], params[1], params[2], params[3])
  time_total = np.max(sequences)

  log_q_z = np.zeros((len(sequences)))

  term_s = sequences[:, 0] >= time_total
  term_i = (sequences[:, 0] + sequences[:, 1] + sequences[:, 2]) >= time_total

  start_s = sequences[:, 0] > 0
  reach_i = (sequences[:, 2] > 0)

  # Terms due to s
  log_q_z += start_s * np.log(a0) + (1-start_s) * np.log(1-a0)
  log_q_z += start_s * (sequences[:, 0]-1)*np.log(feature_weight_0)
  log_q_z += start_s * (1-term_s) * np.log(1-feature_weight_0)

  # Terms due to e
  log_q_z += (1-term_s) * (sequences[:, 1] - 1) * np.log(feature_weight_1)
  log_q_z += reach_i * np.log(1-feature_weight_1)

  # Terms due to I
  log_q_z += reach_i * (sequences[:, 2]-1) * np.log(feature_weight_2)
  log_q_z += reach_i * (1-term_i) * np.log(1-feature_weight_2)

  return log_q_z


def sigmoid(x):
  return 1/(1+np.exp(-x))


def softmax(x):
  y = x - np.max(x)
  return np.exp(y)/np.sum(np.exp(y))


@numba.njit
def precompute_d_penalty_terms_fn(
    q_marginal_infected: np.ndarray,
    p0: float,
    p1: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
  """Precompute penalty terms for inference with Factorised Neighbors.

  Consider similarity to 'precompute_d_penalty_terms_vi' and how the log-term is
  applied.
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term = np.zeros((num_time_steps+1))
  d_no_term = np.zeros((num_time_steps+1))

  if len(past_contacts) == 0:
    return d_term, d_no_term

  assert past_contacts[-1][0] < 0

  # Scales with O(T)
  t_contact = past_contacts[0][0]
  contacts = [np.int32(x) for x in range(0)]
  for row in past_contacts:
    if row[0] == t_contact:
      contacts.append(row[1])
    else:
      # Calculate in log domain to prevent underflow
      log_expectation = 0.

      # Scales with O(num_contacts)
      for user_contact in contacts:
        prob_infected = q_marginal_infected[user_contact][t_contact]
        log_expectation += np.log(prob_infected*(1-p1) + (1-prob_infected))

      d_no_term[t_contact+1] = log_expectation
      d_term[t_contact+1] = (
        np.log(1 - (1-p0)*np.exp(log_expectation)) - np.log(p0))

      # Reset loop stuff
      t_contact = row[0]
      contacts = [np.int32(x) for x in range(0)]

      if t_contact < 0:
        break

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term, d_no_term


@numba.njit
def precompute_d_penalty_terms_fn2(
    q_marginal_infected: np.ndarray,
    p0: float,
    p1: float,
    past_contacts: np.ndarray,
    num_time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
  """Precompute penalty terms for inference with Factorised Neighbors.

  Consider similarity to 'precompute_d_penalty_terms_vi' and how the log-term is
  applied.
  """
  # Make num_time_steps+1 longs, such that penalties are 0 when t0==0
  d_term = np.zeros((num_time_steps+1), dtype=np.float32)
  d_no_term = np.zeros((num_time_steps+1), dtype=np.float32)

  if len(past_contacts) == 0:
    return d_term, d_no_term

  # past_contacts is padded with -1, so break when contact time is negative
  assert past_contacts[-1][0] < 0

  # Scales with O(T)
  log_expectations = np.zeros((num_time_steps+1), dtype=np.float32)
  happened = np.zeros((num_time_steps+1), dtype=np.float32)

  # t_contact = past_contacts[0][0]
  # contacts = [np.int32(x) for x in range(0)]
  for row in past_contacts:
    time_inc = int(row[0])
    if time_inc < 0:
      # past_contacts is padded with -1, so break when contact time is negative
      break

    happened[time_inc+1] = 1
    p_inf_inc = q_marginal_infected[int(row[1])][time_inc]
    log_expectations[time_inc+1] += np.log(p_inf_inc*(1-p1) + (1-p_inf_inc))

  # Additional penalty term for not terminating, negative by definition
  d_no_term = log_expectations
  # Additional penalty term for not terminating, usually positive
  d_term = (np.log(1 - (1-p0)*np.exp(log_expectations)) - np.log(p0))

  # Prevent numerical imprecision error
  d_term *= happened

  # No termination (when t0 == num_time_steps) should not incur penalty
  # because the prior doesn't contribute the p0 factor either
  d_term[num_time_steps] = 0.
  return d_term.astype(np.float32), d_no_term.astype(np.float32)


def it_num_infected_probs(probs: List[float]) -> Iterable[Tuple[int, float]]:
  """Iterates over the number of infected neighbors and its probabilities.

  NOTE: this function scales exponential in the number of neighbrs,
  O(2**len(probs))

  Args:
    probs: List of floats, each being the probability of a neighbor being
    infected.

  Returns:
    iterator with tuples of the number of infected neighbors and its probability
  """
  for is_infected in itertools.product([0, 1], repeat=len(probs)):
    yield sum(is_infected), math.prod(
      abs(is_infected[i] - 1 + probs[i]) for i in range(len(probs)))


def maybe_make_dir(dirname: str):
  if not os.path.exists(dirname):
    logger.info(os.getcwd())
    logger.info(f"Making data_dir {dirname}")
    os.makedirs(dirname)


@numba.njit
def quantize(message: np.ndarray, num_levels: int
             ) -> Union[np.ndarray, float]:
  """Quantizes a message based on rounding.

  Numerical will be mid-bucket.

  TODO: implement quantization with coding scheme.
  """
  dtype_in = message.dtype
  if num_levels < 0:
    return message

  # if np.any(message > 1. + 1E-5):
  #   logger.info(np.min(message))
  #   logger.info(np.max(message))
  #   logger.info(np.mean(message))
  #   raise ValueError(f"Invalid message {message}")
  message = np.minimum(message, 1.-1E-9)
  message_at_floor = np.floor(message * num_levels) / num_levels
  return (message_at_floor + (.5 / num_levels)).astype(dtype_in)


@numba.njit
def quantize_floor(message: Union[np.ndarray, float], num_levels: int
                   ) -> Union[np.ndarray, float]:
  """Quantizes a message based on rounding.

  Numerical will be at the floor of the bucket.

  TODO: implement quantization with coding scheme.
  """
  if num_levels < 0:
    return message

  # if np.any(message > 1. + 1E-5):
  #   logger.info(np.min(message))
  #   logger.info(np.max(message))
  #   logger.info(np.mean(message))
  #   raise ValueError(f"Invalid message {message}")
  return np.floor(message * num_levels) / num_levels


def get_cpu_count() -> int:
  # Divide cpu_count among tasks when running multiple tasks via SLURM
  num_tasks = 1
  if (slurm_ntasks := os.getenv("SLURM_NTASKS")):
    num_tasks = int(slurm_ntasks)
  return int(os.cpu_count() // num_tasks)


@numba.njit
def normalize(x: np.ndarray) -> np.ndarray:
  return x / np.sum(x)


def check_exists(filename: str):
  if not os.path.exists(filename):
    logger.warning(f"File does not exist {filename}, current wd {os.getcwd()}")


def get_stale_users_slice(
    users_stale: np.ndarray,
    user_interval: Tuple[int, int],
    num_users: int) -> Optional[np.ndarray]:
  """Returns a slice of users that are stale.

  Userids in return are relative to the user_interval.

  Args:
    users_stale: np.ndarray of userids that are stale, in absolute userid space.
    user_interval: Tuple of (start, end) of the user interval.
    num_users: Number of users in the entire graph.
  """
  # If None, return None
  if users_stale is None:
    return None

  # If no slicing is happening, return users_stale
  num_users_interval = user_interval[1] - user_interval[0]
  if num_users_interval == num_users:
    return users_stale

  select = np.logical_and(
    users_stale >= user_interval[0],
    users_stale < user_interval[1])
  users_stale = users_stale[select]
  return users_stale - user_interval[0]


def sample_stale_users(
    user_slice: Optional[np.ndarray], num_users: int) -> Optional[np.ndarray]:
  """Samples half of the users in the slice to be stale.

  Args:
    user_slice: Slice of users to sample from, with userid relative to
      user_interval.
    num_users: Number of users in the entire graph.

  Returns:
    Binary vector of length num_users, with 1s indicating users that are stale.
  """
  if user_slice is None:
    return None
  np.random.shuffle(user_slice)
  users_stale_now = user_slice[:int(len(user_slice) // 2)]

  users_stale_binary = np.zeros((num_users,), dtype=np.bool)
  users_stale_binary[users_stale_now] = True
  return users_stale_binary


def get_stale_users_binary(
    user_slice: Optional[np.ndarray], num_users: int) -> np.ndarray:
  """Returns a binary vector of users that are non-stale.

  The vector is 1 for non-stale users, 0 for stale users.
  Args:
    user_slice: Slice of users to sample from, with userid in absolute userid
    num_users: Number of users in the entire graph.

  Returns:
    Binary vector of length num_users, with 0s indicating users that are stale.
  """
  if user_slice is None:
    return np.ones((num_users, 1), dtype=np.bool)

  users_stale_binary = np.ones((num_users, 1), dtype=np.bool)
  users_stale_binary[user_slice] = False
  return users_stale_binary


def get_joblib_backend():
  """Determines a backend for joblib.

  When developing on local PC, then use 'threading'.
  On remote computers, use 'loky'.
  """
  if "carbon" in socket.gethostname():
    return "threading"
  return "loky"


def make_plain_observations(obs):
  return [(o['u'], o['time'], o['outcome']) for o in obs]


def make_plain_contacts(contacts) -> List[constants.Contact]:
  return [
    (c['u'], c['v'], c['time'], int(c['features'][0])) for c in contacts]


def spread_buckets(num_samples: int, num_buckets: int) -> np.ndarray:
  assert num_samples >= num_buckets
  num_samples_per_bucket = (int(np.floor(num_samples / num_buckets))
                            * np.ones((num_buckets)))
  num_remaining = int(num_samples - np.sum(num_samples_per_bucket))
  num_samples_per_bucket[:num_remaining] += 1
  return num_samples_per_bucket


# @functools.lru_cache(maxsize=1)
def spread_buckets_interval(num_samples: int, num_buckets: int) -> np.ndarray:
  num_users_per_bucket = spread_buckets(num_samples, num_buckets)
  return np.concatenate(([0], np.cumsum(num_users_per_bucket)))


@contextlib.contextmanager
def timeit(message: str):
  tstart = time.time()
  yield
  logger.info(f"{message} took {time.time() - tstart:.3f} seconds")


def make_default_array(
    list_of_data: List[Any], dtype, rowlength: int = 3) -> np.ndarray:
  """Makes a default array.

  Args:
    list_of_data: list of data to be put in the array
    dtype: the dtype of the array

  Returns:
    a numpy array with the data in it
  """
  array = np.array(list_of_data, dtype=dtype, ndmin=2)
  if np.prod(array.shape) == 0:
    return -1 * np.ones((0, rowlength), dtype=dtype)

  return array
