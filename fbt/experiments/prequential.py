"""Experiments related to sequential predicton and simulation."""
import datetime
import json
import numba
import numpy as np
from fbt import constants, util
import os
import random
from typing import Any, Dict, Iterable, List, Tuple, Union
import pandas as pd


def dump_results(
    datadir: str, **kwargs):
  fname = os.path.join(datadir, "prec_recall_ir.npz")
  with open(fname, 'wb') as fp:
    np.savez(fp, **kwargs)


def dump_results_json(
    datadir: str,
    cfg: Dict[str, Any],
    **kwargs):
  """Dumps the results of an experiment to JSONlines."""
  kwargs["time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  kwargs["time_day"] = datetime.datetime.now().strftime("%Y%m%d")
  kwargs["slurm_id"] = str(os.getenv('SLURM_JOB_ID'))  # Defaults to 'none'
  kwargs["slurm_name"] = str(os.getenv('SLURM_JOB_NAME'))  # Defaults to 'none'
  kwargs["sweep_id"] = str(os.getenv('SWEEPID'))  # Defaults to 'none'

  model_keys = [
    "p0", "p1", "alpha", "beta", "prob_g", "prob_h",
    "noisy_test", "num_days_window", "quantization",
    "num_rounds", "sib_mult"]
  data_keys = [
    "num_users", "num_time_steps", "num_days_quarantine",
    "fraction_test", "fraction_stale"]

  for key in model_keys:
    kwargs[f"model.{key}"] = cfg["model"][key]

  for key in data_keys:
    kwargs[f"data.{key}"] = cfg["data"][key]

  fname = os.path.join(datadir, "results.jl")
  with open(fname, 'a') as fp:
    fp.write(json.dumps(kwargs) + "\n\r")


def init_states_observations(
    num_users: int, num_days: int
    ) -> Tuple[np.ndarray, List[constants.Observation]]:
  """Initializes the states and observations for prequential simulation."""
  states = np.zeros((num_users, num_days), dtype=np.int16)

  num_users_E_at_start = 2
  observations_all = []

  for user in range(num_users_E_at_start):
    states[user, 0] = 1
    observations_all.append((user, 2, 1))
  return states, observations_all


def simulate_one_day(
    states: np.ndarray, contacts_list: List[constants.Contact], timestep: int,
    p0: float, p1: float, g: float, h: float) -> np.ndarray:
  """Simulates the states for one day, given the contacts."""
  # Sample states at t='timestep' given states up to and including 'timestep-1'

  num_users = states.shape[0]

  # Construct counter on every call, because contacts may change
  infect_counter = util.InfectiousContactCount(
    contacts=contacts_list,
    samples=None,
    num_users=num_users,
    num_time_steps=timestep+1,
  )
  for user in range(num_users):
    if states[user][timestep-1] == 0:
      log_f_term = np.log(1-p0)
      for _, user_u, _ in infect_counter.get_past_contacts_at_time(
          user, timestep-1):
        if states[user_u][timestep-1] == 2:
          log_f_term += np.log(1-p1)
      p_state_up = 1-np.exp(log_f_term)
    elif states[user][timestep-1] == 1:
      p_state_up = g
    elif states[user][timestep-1] == 2:
      p_state_up = h
    elif states[user][timestep-1] == 3:
      p_state_up = 0

    # Increase state according to random sample
    state_up = random.random() < p_state_up
    states[user][timestep] = states[user][timestep-1] + state_up
  return states


@numba.njit
def get_observations_one_day(
    states: np.ndarray,
    users_to_observe: np.ndarray,
    num_obs: int,
    timestep: int,
    p_obs_infected: np.ndarray,
    p_obs_not_infected: np.ndarray,
    obs_rng: np.random._generator.Generator,
    positive_e_state: bool = False) -> np.ndarray:
  """Makes observations for tests on one day.

  Args:
    states: The states of the users, should be in values {0, 1, 2, 3},
      array of length num_users.
    users_to_observe: The users to observe, array of length num_obs.
    num_obs: The number of observations to make.
    timestep: The timestep of the observations.
    p_obs_infected: The probability of a positive test for an infected user.
    p_obs_not_infected: The probability of a positive test for a not infected.
    obs_rng: Random number generator to ensure reproducibility for a fixed seed.
    positive_e_state: Whether to observe E state as positive

  Returns:
    The observations, array of shape (num_obs, 3), where the columns are (user,
      timestep, outcome)
  """
  if num_obs < 1:
    return np.zeros((0, 3), dtype=np.int32)

  assert len(states.shape) == 1

  observations = np.zeros((num_obs, 3), dtype=np.int32)

  assert np.abs(p_obs_infected[0] + p_obs_infected[1] - 1.) < 0.001
  assert np.abs(p_obs_not_infected[0] + p_obs_not_infected[1] - 1.) < 0.001

  states_user = states[users_to_observe]
  positive = states_user == 2

  if positive_e_state:
    positive = np.logical_or(positive, states_user == 1)

  sample_prob = np.where(positive, p_obs_infected[1], p_obs_not_infected[1])

  assert sample_prob.shape == (num_obs, )

  observations[:, 0] = users_to_observe
  observations[:, 1] = timestep
  observations[:, 2] = sample_prob >= obs_rng.random(num_obs)

  return observations.astype(np.int32)


def calc_prec_recall(
    states: np.ndarray, users_to_quarantine: np.ndarray) -> Tuple[float, float]:
  """Calculates precision and recall for quarantine assignments.

  Note that when no users are in state E or I, the recall is 1.0 (achieved by
  seting epsilon to a small number).
  """
  assert len(states.shape) == 1
  eps = 1E-9  # Small number to avoid division by zero

  users_quarantine_array = np.zeros_like(states)
  users_quarantine_array[users_to_quarantine] = 1.
  states_e_i = np.logical_or(
    states == 1,
    states == 2,
  )

  true_positives = np.sum(
    np.logical_and(states_e_i, users_quarantine_array)) + eps

  precision = true_positives / (np.sum(users_quarantine_array) + eps)
  recall = true_positives / (np.sum(states_e_i) + eps)
  return precision, recall


def remove_quarantine_users(
    contacts: List[constants.Contact],
    quarantine_users: Union[List[int], np.ndarray],
    t_start: int,
    t_delta: int) -> List[constants.Contact]:
  """Removes quarantined users from contact list.

  A contact will be removed if EITHER the user u or user v is being quarantined
  and the timestep is GREATER OR EQUAL t_start and LESS THAN t_start+t_delta.
  """
  def filter_func(contact):
    return not (
      ((contact[0] in quarantine_users) or (contact[1] in quarantine_users))
      and (contact[2] >= t_start)
      and (contact[2] < (t_start+t_delta)))
  return list(filter(filter_func, contacts))


def get_evidence_obs(
    observations: constants.ObservationList,
    z_states: np.ndarray,
    alpha: float,
    beta: float) -> float:
  """Calculates evidence for the observations, integrating out the states."""
  p_obs_infected = [alpha, 1-alpha]
  p_obs_not_infected = [1-beta, beta]

  log_like = 0.
  for obs in observations:
    user, timestep, outcome = obs[0], obs[1], obs[2]
    p_inf = z_states[user, timestep, 2]

    log_like += np.log(
      p_inf*p_obs_infected[outcome] + (1-p_inf)*p_obs_not_infected[outcome]
      + 1E-9)
  return log_like


def decide_tests(
    scores_infect: np.ndarray,
    num_tests: int,
    rand_gen: np.random._generator.Generator = np.random.default_rng(),
    pop_info: pd.core.frame.DataFrame = pd.DataFrame(),
    policy: str = "combi",
    stochastic: bool = False,
    hybrid_split: float = 0.5,
    feature_weight_0: float = 1.0,
    feature_weight_1: float = 1.0,
    feature_weight_2: float = 1.0,
    feature_pow_1: float = 1.0,
    feature_pow_2: float = 1.0,
    degrees: np.ndarray = np.array([])) -> np.ndarray:
  """Returns the people who should be tested according to a given policy."""
  assert num_tests < len(scores_infect)

  if num_tests == 0:
    return np.array([]).astype(np.int32)

  if policy == "hybrid":
    # Policy where (deterministic) part of tests is used on highest risk score,
    # and remaining (stochastic) part is used randomly on rest of population
    sto_part, det_part = np.split(
      np.argsort(scores_infect),
      [len(scores_infect)-int(hybrid_split*num_tests)])

    users_to_test = np.append(
      det_part, rand_gen.choice(sto_part,
                                num_tests - int(hybrid_split*num_tests),
                                replace=False))

    return users_to_test.astype(np.int32)

  if "combi" in policy:
    # Policy that is a weighted combination of multiple features
    features = policy.split(',')
    features.remove('combi')

    if "norm_score" in features:
      features.remove("norm_score")
      if np.percentile(scores_infect, 80) > 0:
        scores_infect = scores_infect/np.percentile(scores_infect, 80)

    exp = False
    if "exp" in features:
      features.remove("exp")
      exp = True

    scores = []

    for i, score_infect in enumerate(scores_infect):
      feature_values = [0,0]
      for j, feature in enumerate(features):
        if feature == 'deg':
          feature_list = degrees
        else:
          feature_list = pop_info[feature].to_numpy()

        feature_values[j] = feature_list[i]/np.percentile(feature_list, 80)

      if exp:
        scores.append(
          feature_weight_0*score_infect
          + (feature_weight_1*np.exp(feature_values[0]))**feature_pow_1
          + (feature_weight_2*np.exp(feature_values[1]))**feature_pow_2)
      else:
        scores.append(
          feature_weight_0*score_infect
          + (feature_weight_1*feature_values[0])**feature_pow_1
          + (feature_weight_2*feature_values[1])**feature_pow_2)
  else:
    raise ValueError((
      f"Not recognised testing policy {policy}. Should be one of"
      f"['hybrid', 'combi']"
    ))

  if stochastic:
    # The "stochastic" variable also works as temperature
    scores = np.power(scores, stochastic)

    # Probabilities cannot be below zero
    if min(scores) < 0:
      scores -= min(scores)

    # Prevent divide by zero error
    if scores.sum() == 0:
      scores = np.ones(len(scores_infect))
    scores /= scores.sum()

    # Prevent error if there aren't enough nonzero scores
    if num_tests > np.count_nonzero(scores):
      # Add all users with nonzero probability, randomly sample those with zero
      users_to_test = np.array(np.nonzero(scores)[0])
      users_to_test = np.append(
        users_to_test, rand_gen.choice(
          np.where(scores == 0)[0],
          num_tests-np.count_nonzero(scores),
          replace=False))
    else:
      users_to_test = rand_gen.choice(
        np.arange(len(scores_infect)), num_tests, p=scores, replace=False)
  else:
    users_to_test = np.argsort(scores)[-num_tests:]

  return users_to_test.astype(np.int32)


def delay_contacts(
    contacts: List[constants.Contact], delay: int
    ) -> Iterable[constants.Contact]:
  """Offsets the contacts or observations by a number of days."""
  for contact in contacts:
    yield (contact[0], contact[1], contact[2] + delay, contact[3])


def offset_observations(
    observations_in: constants.ObservationList, offset: int
    ) -> constants.ObservationList:
  """Offsets the cobservations by a number of days."""
  if offset == 0:
    return observations_in

  observations = np.copy(observations_in)
  observations = observations[observations[:, 1] >= offset]
  observations[:, 1] -= offset
  return observations


def offset_contacts(
    contacts: Iterable[constants.Contact], offset: int
    ) -> Iterable[constants.Contact]:
  """Offsets the cobservations by a number of days."""
  if offset == 0:
    yield from contacts
    return

  for contact in contacts:
    if contact[2] >= offset:
      yield (contact[0], contact[1], contact[2] - offset, contact[3])
