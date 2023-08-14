"""Tests for prequential.py"""
import numpy as np
from fbt.config import config
from fbt.experiments import prequential


def test_simulate_one_day():
  contacts_all = [
    (0, 1, 2, 1),
  ]
  states = np.array([
    [2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0]
  ])

  states_new = prequential.simulate_one_day(
    states=states,
    contacts_list=contacts_all,
    timestep=3,
    p0=0.001,
    p1=1.,
    g=1/3,
    h=1/3)

  np.testing.assert_almost_equal(states_new[1][3], 1.)


def test_get_observations_one_day():

  p_inf = np.array([0., 1.])
  p_ninf = np.array([1., 0.])

  states = np.array([0, 0, 2, 0])
  users_to_observe = np.array([0, 1, 2, 3], dtype=np.int32)
  observations = list(prequential.get_observations_one_day(
    states=states, users_to_observe=users_to_observe, num_obs=4, timestep=1,
    p_obs_infected=p_inf, p_obs_not_infected=p_ninf,
    obs_rng=np.random.default_rng()))

  observations_expected = [
    (0, 1, 0),
    (1, 1, 0),
    (2, 1, 1),
    (3, 1, 0),
  ]

  np.testing.assert_array_almost_equal(observations, observations_expected)

  # Flip observation model
  states = np.array([0, 0, 2, 0])
  observations = list(prequential.get_observations_one_day(
    states=states, users_to_observe=users_to_observe, num_obs=4, timestep=1,
    p_obs_infected=p_ninf, p_obs_not_infected=p_inf,
    obs_rng=np.random.default_rng()))

  observations_expected = [
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (3, 1, 1),
  ]

  np.testing.assert_array_almost_equal(observations, observations_expected)


def test_get_observations_one_day_slice():

  p_inf = np.array([0., 1.])
  p_ninf = np.array([1., 0.])

  states = np.array([0, 0, 2, 0, 0, 0])
  users_to_observe = np.array([3, 1, 4, 2], dtype=np.int32)
  observations = list(prequential.get_observations_one_day(
    states=states,
    users_to_observe=users_to_observe,
    num_obs=len(users_to_observe),
    timestep=1,
    p_obs_infected=p_inf,
    p_obs_not_infected=p_ninf,
    obs_rng=np.random.default_rng()))

  observations_expected = [
    (3, 1, 0),
    (1, 1, 0),
    (4, 1, 0),
    (2, 1, 1),
  ]

  np.testing.assert_array_almost_equal(observations, observations_expected)


def test_get_observations_one_day_rng_seed():

  test_rng = np.random.default_rng(seed=25)
  p_inf = np.array([0.2, 0.8])
  p_ninf = np.array([0.8, 0.2])

  states = np.array([0, 0, 2, 0])
  users_to_observe = np.array([0, 1, 2, 3], dtype=np.int32)
  observations = list(prequential.get_observations_one_day(
    states=states, users_to_observe=users_to_observe, num_obs=4, timestep=1,
    p_obs_infected=p_inf, p_obs_not_infected=p_ninf, obs_rng=test_rng))

  observations_expected = [
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 1),
    (3, 1, 0),
  ]

  np.testing.assert_array_almost_equal(observations, observations_expected)

  # Flip observation model
  states = np.array([0, 0, 2, 0])
  observations = list(prequential.get_observations_one_day(
    states=states, users_to_observe=users_to_observe, num_obs=4, timestep=1,
    p_obs_infected=p_ninf, p_obs_not_infected=p_inf, obs_rng=test_rng))

  observations_expected = [
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (3, 1, 1),
  ]

  np.testing.assert_array_almost_equal(observations, observations_expected)


def test_get_observations_one_day_slice_rng_seed():

  p_inf = np.array([0.2, 0.8])
  p_ninf = np.array([0.8, 0.2])

  states = np.array([0, 0, 2, 0, 0, 0])
  users_to_observe = np.array([3, 1, 4, 2], dtype=np.int32)
  observations = list(prequential.get_observations_one_day(
    states=states,
    users_to_observe=users_to_observe,
    num_obs=len(users_to_observe),
    timestep=1,
    p_obs_infected=p_inf,
    p_obs_not_infected=p_ninf,
    obs_rng=np.random.default_rng(seed=25)))

  observations_expected = [
    (3, 1, 1),
    (1, 1, 1),
    (4, 1, 0),
    (2, 1, 1),
  ]

  np.testing.assert_array_almost_equal(observations, observations_expected)


def test_calc_prec_recall():
  states = np.array([0, 0, 1, 1, 2, 2, 3, 4, 3])
  users_to_quarantine = np.array([1, 2, 3, 5, 7])
  precision, recall = prequential.calc_prec_recall(states, users_to_quarantine)

  np.testing.assert_almost_equal(precision, 0.6)
  np.testing.assert_almost_equal(recall, 0.75)


def test_calc_prec_recall_nan():
  states = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4])
  users_to_quarantine = np.array([1, 2, 3, 5, 7])
  precision, recall = prequential.calc_prec_recall(states, users_to_quarantine)

  # When no users are infected, precision should be 0 and recall should be 1
  np.testing.assert_almost_equal(precision, 0.0)
  np.testing.assert_almost_equal(recall, 1.0)


def test_get_evidence_obs():
  alpha = 1E-9
  beta = 1E-9
  observations = [
    (0, 1, 1),
    (1, 1, 1)]

  z_states = np.array([
    [[0., 0., 1., 0.], [0., 0., 1., 0.]],
    [[0., 0., 1., 0.], [0., 0., 1., 0.]]])

  # First test with negligible alpha, beta
  log_like = prequential.get_evidence_obs(observations, z_states, alpha, beta)
  np.testing.assert_almost_equal(log_like, 0.0)

  # Then test with arbitrary alpha, beta
  alpha = 0.1
  log_like = prequential.get_evidence_obs(observations, z_states, alpha, beta)
  np.testing.assert_almost_equal(log_like, 2*np.log(1.-alpha))

  # Then test with non-unit infectiousness prediction
  z_states = np.array([
    [[0., 0., 1., 0.], [0.1, 0.1, .8, 0.]],
    [[0., 0., 1., 0.], [0.1, 0.1, .8, 0.]]])
  log_like = prequential.get_evidence_obs(observations, z_states, alpha, beta)
  np.testing.assert_almost_equal(log_like, 2*np.log(.8 * (1.-alpha) + .2*beta))


def test_decide_tests():

  scores = np.array([.9, .9, .3, .4, .5, .1, .2])
  users_to_test = prequential.decide_tests(
    scores, num_tests=3)

  np.testing.assert_array_almost_equal(
    np.sort(users_to_test), np.array([0, 1, 4]))
  assert users_to_test.dtype == np.int32


def test_offset_observations():
  observations = np.array([
    (0, 11, 1),
    (1, 11, 1),
    (2, 1, 1),
    ], dtype=np.int32)
  observations_new = prequential.offset_observations(observations, offset=2)

  observations_expected = [
    (0, 9, 1),
    (1, 9, 1),
    ]

  np.testing.assert_array_almost_equal(observations_new, observations_expected)

  assert observations[0][1] == 11, (
    "Time changed. Did the function change value in place (change value instead"
    "of change reference. For now the code is based on a separate data "
    "structure with absolute timing"
    + f"{observations[0][1]} does not match 11")


def test_offset_observations_empty():
  observations = np.zeros((0, 3), dtype=np.int32)
  observations_new = prequential.offset_observations(observations, offset=2)

  assert len(observations_new) == 0


def test_dump_results():

  fname_config_data = "fbt/config/small_graph_01.ini"
  fname_config_model = "fbt/config/model_IG02.ini"

  config_data = config.ConfigBase(fname_config_data).to_dict()
  config_model = config.ConfigBase(fname_config_model).to_dict()

  cfg = {
    "data": config_data,
    "model": config_model,
  }

  prequential.dump_results_json(
    datadir="results/tmp/",
    cfg=cfg,
    some_result=np.random.randn(13).tolist()
  )
