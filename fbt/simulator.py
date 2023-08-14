"""Simulating individuals in a pandemic with SEIR states."""
from abc import ABC
from COVID19 import model as abm_model
from COVID19 import simulation
import covid19

from fbt import constants, logger, util
from fbt.experiments import prequential
import numpy as np
import os
from typing import Any, Dict, List, Union
import pandas as pd


def _embed_contact(contact_tuple) -> constants.Contact:
  # Tuples from ABM simulator have semantics:
  # (user_from, user_to, timestep, features)
  # TODO replace with 'contact_tuple[3]'
  return (contact_tuple[0], contact_tuple[1], contact_tuple[2], 1)


class Simulator(ABC):
  """Base class for a simulator."""

  def __init__(
      self,
      num_time_steps: int,
      num_users: int,
      params: Dict[str, Any],
      positive_e_state: bool = False,
      rng_seed: int = 123) -> None:
    self.num_time_steps = num_time_steps
    self.num_users = num_users
    self.params = params
    self.rng_seed = rng_seed

    self._day_current = 0
    self.states = None  # Type will depend on implementation

    # When true, both E and I state are considered positive
    self._postive_e_state = positive_e_state

    # Note that contacts are offset with self._day_start_window and contacts
    # prior to self._day_start_window have been discarded.
    self._day_start_window = 0
    # Array with rows (user, timestep, outcome), all integers
    self._observations_all = np.zeros((0, 3), dtype=np.int32)
    # List of (user_u, user_v, timestep, [features])
    self._contacts = []

  def set_window(self, days_offset: int):
    """Sets the window with days_offset at day0.

    All days will start counting 0 at days_offset.
    The internal counter self._day_start_window keeps track of the previous
    counting for day0.
    """
    to_cut_off = max((0, days_offset - self._day_start_window))
    self._observations_all = prequential.offset_observations(
      self._observations_all, to_cut_off)
    self._contacts = list(prequential.offset_contacts(
      self._contacts, to_cut_off))
    self._day_start_window = days_offset

  def get_current_day(self) -> int:
    """Returns the current day in absolute counting.

    Note, this day number is INDEPENDENT of the windowing.
    """
    # 0-based indexes!
    return self._day_current

  def get_states_today(self) -> np.ndarray:
    """Returns the states an np.ndarray in size [num_users].

    Each element in [0, 1, 2, 3].
    """
    return np.zeros((self.num_users), dtype=np.int32)

  def get_contacts(self) -> List[constants.Contact]:
    """Returns contacts.

    Note that contacts are offset with self._day_start_window and contacts prior
    to self._day_start_window have been discarded.
    """
    return self._contacts

  def get_observations_today(
      self,
      users_to_observe: np.ndarray,
      p_obs_infected: np.ndarray,
      p_obs_not_infected: np.ndarray,
      obs_rng: np.random._generator.Generator,
      ) -> constants.ObservationList:
    """Returns the observations for current day."""
    assert users_to_observe.dtype == np.int32

    day_relative = self.get_current_day() - self._day_start_window
    observations_new = prequential.get_observations_one_day(
      self.get_states_today(),
      users_to_observe,
      len(users_to_observe),
      day_relative,
      p_obs_infected,
      p_obs_not_infected,
      obs_rng,
      positive_e_state=self._postive_e_state)
    self._observations_all = np.concatenate(
      (self._observations_all, observations_new), axis=0)
    return observations_new

  def get_observations_all(self) -> constants.ObservationList:
    """Returns all observations."""
    return self._observations_all

  def step(self, num_steps: int = 1):
    """Advances the simulator by num_steps days."""
    self._day_current += num_steps

  def quarantine_users(
      self,
      users_to_quarantine: Union[np.ndarray, List[int]],
      num_days: int):
    """Quarantines the defined users.

    This function will remove the contacts that happen TODAY (and which may
    spread the virus and cause people to shift to E-state tomorrow).
    """


class DummySimulator(Simulator):
  """Simulator with dummy functions."""

  def init_day0(self):
    """Initializes the simulator for day0."""
    self._contacts = []
    self.states = np.zeros(
      (self.num_users, self.num_time_steps, 4), dtype=np.uint8)
    self._observations_all = []


class CRISPSimulator(Simulator):
  """Simulator based on generative code with CRISP paper."""

  def init_day0(self, contacts: List[constants.Contact]):
    """Initializes the simulator for day0."""
    self._contacts = contacts
    states, obs_list_current = prequential.init_states_observations(
      self.num_users, self.num_time_steps)

    self.states = states  # np.ndarray in size [num_users, num_timesteps]
    self._observations_all = np.array(obs_list_current, dtype=np.int32)

  def get_states_today(self) -> np.ndarray:
    """Returns the states an np.ndarray in size [num_users].

    Each element in [0, 1, 2, 3].
    """
    return self.states[:, self._day_current]

  def step(self, num_steps: int = 1):
    """Advances the simulator by num_steps days."""
    self._day_current += num_steps

    # Set contact days as simulate_one_day works with absolute times.
    contacts = list(
      prequential.delay_contacts(self._contacts, self._day_start_window))
    self.states = prequential.simulate_one_day(
      self.states, contacts, self._day_current,
      self.params["p0"], self.params["p1"], self.params["g"], self.params["h"])

    assert np.sum(self.states[:, self._day_current+1:]) == 0

  def quarantine_users(
      self,
      users_to_quarantine: Union[np.ndarray, List[int]],
      num_days: int):
    """Quarantines the defined users.

    This function will remove the contacts that happen TODAY (and which may
    spread the virus and cause people to shift to E-state tomorrow).
    """
    # Make quarantines relative to window
    start_quarantine = self.get_current_day() + 1 - self._day_start_window

    self._contacts = prequential.remove_quarantine_users(
      self._contacts, users_to_quarantine, start_quarantine,
      t_delta=num_days)


class ABMSimulator(Simulator):
  """Simulator based on Oxford ABM."""

  def __init__(
      self,
      num_time_steps: int,
      num_users: int,
      params: Dict[str, Any],
      positive_e_state: bool = False,
      rng_seed: int = 123,
      ) -> None:
    super().__init__(
      num_time_steps, num_users, params, positive_e_state=positive_e_state,
      rng_seed=rng_seed)

    filename = "baseline_parameters.csv"
    filename_hh = "baseline_household_demographics.csv"

    input_param_file = os.path.join(constants.ABM_HOME, filename)
    input_households = os.path.join(constants.ABM_HOME, filename_hh)

    util.check_exists(input_param_file)
    util.check_exists(input_households)
    util.maybe_make_dir("results/tmp/")

    logger.info("Construct ABM simulator")
    params = abm_model.Parameters(  # TODO, import full module!
      input_param_file=input_param_file,
      param_line_number=1,
      output_file_dir="results/tmp/",
      input_households=input_households
    )

    if num_users < 10000:
      # TODO figure out why this fails
      logger.debug('ABM simulator might fail with <10k users')

    # Start with sufficient amount of initial infections. Start in E-state
    n_seed = 5
    if 20000 < num_users < 200000:
      n_seed = 25
    if num_users >= 200000:
      n_seed = 50

    params.set_param("n_total", num_users)
    params.set_param("n_seed_infection", n_seed)
    params.set_param("days_of_interactions", 7)
    params.set_param("rng_seed", rng_seed)

    model_init = abm_model.Model(params)
    self.model = simulation.COVID19IBM(model=model_init)
    self.sim = simulation.Simulation(env=self.model, end_time=num_time_steps)
    logger.info("Finished constructing ABM simulator")

  def init_day0(self, contacts: Any):
    """Initializes the simulator for day0."""
    del contacts

  def get_states_today(self) -> np.ndarray:
    """Returns the states an np.ndarray in size [num_users].

    Each element in [0, 1, 2, 3].
    """
    return np.take(
      constants.state_to_seir,
      np.array(covid19.get_state(self.model.model.c_model)))

  def get_states_abm(self) -> np.ndarray:
    """Returns the states of the underlying abm simulator."""
    return np.array(covid19.get_state(self.model.model.c_model))

  def get_contacts_daily(self, offset=1):
    """Returns a list of the contacts that occured on the past day."""
    return list(map(
      _embed_contact, covid19.get_contacts_daily(
        self.model.model.c_model, self._day_current-offset)))

  def step(self, num_steps: int = 1):
    """Advances the simulator by num_steps days.

    Contacts from OpenABM will be appended to the list of contacts.
    """
    self.sim.steps(num_steps)
    contacts_incoming = self.get_contacts_daily(0)

    self._contacts += prequential.offset_contacts(
      contacts_incoming, self._day_start_window)
    self._day_current += 1

  def quarantine_users(
      self,
      users_to_quarantine: Union[np.ndarray, List[int]],
      num_days: int):
    """Quarantines the defined users.

    This function will remove the contacts that happen TODAY (and which may
    spread the virus and cause people to shift to E-state tomorrow).
    """
    # Timestep of the actual ABM simulator could be found at
    #   * self.model.model.c_model.time
    status = covid19.intervention_quarantine_list(
      self.model.model.c_model,
      list(users_to_quarantine),
      self.get_current_day()+1 + num_days)
    assert int(status) == 0

  def get_pop_info(self):
    return pd.DataFrame(data={
      'state': self.get_states_today(),
      'age': covid19.get_age(self.model.model.c_model),
      'house': covid19.get_house(self.model.model.c_model)})
