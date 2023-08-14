"""Unit tests for simulator.py"""
import numpy as np
from fbt import simulator


def test_window_cut():

  contacts = [
    (0, 23, 0, 1),
    (0, 23, 1, 1),
    (0, 23, 2, 1),
    (0, 23, 3, 1),
    (0, 23, 4, 1),
    (0, 23, 5, 1),
    (0, 23, 6, 1),
    (0, 23, 7, 1),
    (0, 23, 8, 1),
    (0, 23, 9, 1),
  ]

  sim = simulator.CRISPSimulator(num_time_steps=30, num_users=100, params={})
  sim.init_day0(contacts=contacts)

  assert sim.get_contacts() == contacts
  assert len(sim.get_contacts()) == 10

  sim.set_window(days_offset=1)

  assert sim.get_contacts() == contacts[:-1]
  assert len(sim.get_contacts()) == 9

  sim.set_window(days_offset=5)

  assert sim.get_contacts() == contacts[:-5]
  assert len(sim.get_contacts()) == 5

  sim.set_window(days_offset=6)

  assert sim.get_contacts() == contacts[:-6]
  assert len(sim.get_contacts()) == 4

  # Test right away for empty observations
  p_inf = np.array([0., 1.])
  p_ninf = np.array([1., 0.])

  sim.set_window(days_offset=0)
  sim.get_observations_today(
    users_to_observe=np.array([], dtype=np.int32),
    p_obs_infected=p_inf, p_obs_not_infected=p_ninf,
    obs_rng=np.random.default_rng(seed=30))

  # Test for removing all contacts/observations
  assert len(sim.get_contacts()) == 4
  sim.set_window(days_offset=999)
  assert len(sim.get_contacts()) == 0
