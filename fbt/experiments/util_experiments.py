"""Utility functions for running experiments."""
import numpy as np
from fbt import constants, inference, logger
import subprocess
from typing import Any, Dict, Optional


def wrap_fact_neigh_inference(
    num_users: int,
    alpha: float,
    beta: float,
    p0: float,
    p1: float,
    g_param: float,
    h_param: float,
    quantization: int = -1,
    ):
  """Wraps the inference function that runs Factorised Neighbors."""

  def fact_neigh_wrapped(
      observations_list: constants.ObservationList,
      contacts_list: constants.ContactList,
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:

    traces_per_user_fn = inference.fact_neigh(
      num_users=num_users,
      num_time_steps=num_time_steps,
      observations_all=observations_list,
      contacts_all=contacts_list,
      alpha=alpha,
      beta=beta,
      probab_0=p0,
      probab_1=p1,
      g_param=g_param,
      h_param=h_param,
      start_belief=start_belief,
      quantization=quantization,
      users_stale=users_stale,
      num_updates=num_updates,
      verbose=False,
      diagnostic=diagnostic)
    return traces_per_user_fn
  return fact_neigh_wrapped


def wrap_dummy_inference(
    num_users: int,):
  """Wraps the inference function for dummy inference."""

  def dummy_wrapped(
      observations_list: constants.ObservationList,
      contacts_list: constants.ContactList,
      num_updates: int,
      num_time_steps: int,
      start_belief: Optional[np.ndarray] = None,
      users_stale: Optional[np.ndarray] = None,
      diagnostic: Optional[Any] = None) -> np.ndarray:
    del diagnostic, start_belief, num_updates, contacts_list, observations_list
    del users_stale

    predictions = np.random.randn(num_users, num_time_steps, 4)
    predictions /= np.sum(predictions, axis=-1, keepdims=True)

    return predictions

  return dummy_wrapped


def set_noisy_test_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
  """Sets the noise parameters of the observational model."""
  noise_level = cfg["model"]["noisy_test"]
  assert noise_level <= 4

  if noise_level < 0:
    return cfg

  alpha_betas = [
    (1E-9, 1E-9), (.001, 0.01), (.01, .1), (.03, .25), (.5, .5)]

  # Set model parameters
  cfg["model"]["alpha"] = alpha_betas[noise_level][0]
  cfg["model"]["beta"] = alpha_betas[noise_level][1]

  # Don't assume model misspecification, set data parameters the same
  cfg["data"]["alpha"] = alpha_betas[noise_level][0]
  cfg["data"]["beta"] = alpha_betas[noise_level][1]

  return cfg


def make_git_log():
  """Logs the git diff and git show.

  Note that this function has a general try/except clause and will except most
  errors produced by the git commands.
  """
  try:
    result = subprocess.run(
      ['git', 'show', '--summary'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git show \n{result.stdout.decode('utf-8')}")

    result = subprocess.run(['git', 'diff'], stdout=subprocess.PIPE, check=True)
    logger.info(f"Git diff \n{result.stdout.decode('utf-8')}")
  except Exception as e:  # pylint: disable=broad-except
    logger.info(f"Git log not printed due to {e}")
