"""Test functions for util.py."""

from fbt import util
import numpy as np
import random


def test_state_time():
  result = util.state_at_time([6, 0, 0], 4)
  assert float(result.flatten()[0]) == 0, f"result {result}"

  result = util.state_at_time([[1, 4, 5], [1, 3, 5], [1, 2, 5]], 4)
  assert result.tolist() == [1, 2, 2], f"result {result}"

  result = util.state_at_time([[1, 4, 5], [1, 3, 5], [1, 2, 5]], 1)
  assert result.tolist() == [1, 1, 1], f"result {result}"

  result = util.state_at_time([1, 1, 1], 4)
  assert float(result.flatten()[0]) == 3, f"result {result}"


def test_state_time_cache():
  result = util.state_at_time_cache(6, 0, 0, 4)
  assert result == 0

  result = util.state_at_time_cache(1, 4, 5, 4)
  assert result == 1

  result = util.state_at_time_cache(1, 3, 5, 4)
  assert result == 2

  result = util.state_at_time_cache(1, 2, 5, 4)
  assert result == 2

  result = util.state_at_time_cache(1, 2, 5, 1)
  assert result == 1

  result = util.state_at_time_cache(1, 2, 5, 8)
  assert result == 3


def test_calculate_log_c_z():
  num_time_steps = 4

  observations_all = np.array([
    (1, 2, 1),
    (2, 3, 0),
    ])

  a, b = .1, .2

  obs_array = util.make_inf_obs_array(int(num_time_steps), a, b)
  result = util.calc_c_z_u(
    user_interval=(0, 3),
    obs_array=obs_array,
    observations=observations_all)
  expected = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [np.log(b), np.log(b), np.log(b), np.log(1-a), np.log(1-a), np.log(b),
     np.log(1-a), np.log(1-a), np.log(1-a), np.log(1-a), np.log(b), np.log(b),
     np.log(1-a), np.log(1-a), np.log(b), np.log(b), np.log(b), np.log(b),
     np.log(b), np.log(b)],
    [np.log(1-b), np.log(1-b), np.log(1-b), np.log(1-b), np.log(a), np.log(1-b),
     np.log(1-b), np.log(a), np.log(1-b), np.log(a), np.log(a), np.log(1-b),
     np.log(1-b), np.log(a), np.log(a), np.log(1-b), np.log(a), np.log(1-b),
     np.log(1-b), np.log(1-b)]])

  assert result.shape == expected.shape, (
    f"Shapes dont match: {result.shape} {expected.shape}")

  np.testing.assert_array_almost_equal(result, expected)


def test_generate_sequences():
  result = util.generate_sequence_days(time_total=4)
  expected = [(4, 0, 0), (3, 1, 0), (2, 1, 1), (2, 2, 0), (1, 1, 1), (1, 1, 2),
              (1, 2, 1), (1, 3, 0)]

  for x, y in zip(result, expected):
    assert x == y, f"{x} and {y} do not match"


def test_calc_log_a():
  potential_sequences = list(util.generate_sequence_days(time_total=4))
  seq_array = np.stack(potential_sequences, axis=0)

  g = 1 / 8
  h = 1 / 8
  p0 = 0.01

  log_A = util.calc_log_a_start(seq_array, p0, g, h)

  expected = [
    3*np.log(1-p0),
    2*np.log(1-p0),
    np.log(1-p0) + np.log(g),
    np.log(1-p0) + np.log(1-g),
    np.log(g) + np.log(h),
    np.log(g) + np.log(1-h),
    np.log(1-g) + np.log(g),
    2*np.log(1-g)
  ]
  np.testing.assert_array_almost_equal(log_A, expected)


def test_state_seq_to_time_seq():
  seq_days = list(util.generate_sequence_days(4))
  result = util.state_seq_to_time_seq(np.array(seq_days), 5)

  expected = [
    [0, 0, 0, 0, 3],
    [0, 0, 0, 1, 3],
    [0, 0, 1, 2, 3],
    [0, 0, 1, 1, 3],
    [0, 1, 2, 3, 3],
    [0, 1, 2, 2, 3],
    [0, 1, 1, 2, 3],
    [0, 1, 1, 1, 3]]

  np.testing.assert_array_almost_equal(result, np.array(expected))

  seq_days = list(util.generate_sequence_days(4))
  result = util.state_seq_to_hot_time_seq(np.array(seq_days), 5)
  np.testing.assert_array_almost_equal(result.shape, [8, 5, 4])


def test_iter_sequences():
  num_time_steps = 7
  num_seqs = len(list(util.iter_sequences(num_time_steps, start_se=True)))
  np.testing.assert_almost_equal(num_seqs, 64)

  # num_time_steps+1 more
  num_seqs = len(list(util.iter_sequences(num_time_steps, start_se=False)))
  np.testing.assert_almost_equal(num_seqs, 72)


def test_enumerate_log_prior_values_full_sums_1():
  num_time_steps = 7
  p0, g, h = 0.01, 0.2, 0.16

  seqs = np.stack(list(
    util.iter_sequences(time_total=num_time_steps, start_se=False)))

  log_p = util.enumerate_log_prior_values(
    [1-p0, p0, 0., 0.], [1-p0, 1-g, 1-h], seqs, num_time_steps)

  np.testing.assert_almost_equal(np.sum(np.exp(log_p)), 1.0, decimal=3)


def test_infect_contact_count():
  contacts_all = [
    (0, 1, 6, 1),
    (1, 0, 6, 1),
    (0, 2, 5, 1),
    (2, 0, 5, 1),
    (2, 3, 5, 1),
    (3, 2, 5, 1),
    (2, 4, 6, 1),
    (4, 2, 6, 1),
    (5, 0, 6, 1),
    ]

  samples_current = {
    # Being: t0, de, di
    0: [1, 1, 9],
    1: [1, 1, 9],
    2: [1, 1, 9],
    3: [1, 1, 9],
    4: [1, 1, 9],
    5: [1, 1, 9],
    }

  counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=samples_current,
    num_users=6,
    num_time_steps=12)
  result = [counter.num_inf_contacts(2, t) for t in range(4, 8)]
  np.testing.assert_array_almost_equal(result, [0, 0, 2, 1])

  counter.update_infect_count(3, [1, 1, 9], remove=True)
  result = [counter.num_inf_contacts(2, t) for t in range(4, 8)]
  np.testing.assert_array_almost_equal(result, [0, 0, 1, 1])

  counter.update_infect_count(3, [1, 2, 9], remove=False)
  result = [counter.num_inf_contacts(2, t) for t in range(4, 8)]
  np.testing.assert_array_almost_equal(result, [0, 0, 2, 1])

  expected_future_contacts = [(5, 2, 1), (6, 1, 1)]
  result = list(counter.get_future_contacts(0))
  assert expected_future_contacts == result, "Future contacts don't match."

  expected_past_contacts = [(5, 2, 1), (6, 1, 1), (6, 5, 1)]
  result = list(counter.get_past_contacts(0))
  assert expected_past_contacts == result, "Past contacts don't match."


def test_gather_infected_precontacts():
  num_time_steps = 8
  contacts_all = [
    (0, 1, 6, 1),
    (1, 0, 6, 1),
    (0, 2, 5, 1),
    (2, 0, 5, 1),
    (2, 3, 5, 1),
    (3, 2, 5, 1),
    (2, 4, 6, 1),
    (4, 2, 6, 1),
    (5, 0, 6, 1),
    ]

  samples_current = {
    # Being: t0, de, di
    0: [1, 1, 9],
    1: [1, 1, 9],
    2: [1, 1, 9],
    3: [1, 1, 9],
    4: [1, 1, 9],
    5: [1, 1, 9],
    }

  counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=samples_current,
    num_users=6,
    num_time_steps=num_time_steps)

  result = util.gather_infected_precontacts(
    num_time_steps=num_time_steps,
    samples_current=samples_current,
    past_contacts=counter.get_past_contacts(user=0))

  expected = np.array([0, 0, 0, 0, 0, 1, 2, 0])
  np.testing.assert_array_almost_equal(result, expected)

  # Change state of contact by 1 to time different time step
  samples_current[1] = [11, 0, 0]

  result = util.gather_infected_precontacts(
    num_time_steps=num_time_steps,
    samples_current=samples_current,
    past_contacts=counter.get_past_contacts(user=0))

  expected = np.array([0, 0, 0, 0, 0, 1, 1, 0])
  np.testing.assert_array_almost_equal(result, expected)


def test_d_penalty_term():
  contacts_all = np.array([
    (0, 1, 2, 1),
    (1, 0, 2, 1),
    (3, 2, 2, 1),
    (2, 3, 2, 1),
    (4, 5, 2, 1),
    (5, 4, 2, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 5

  infect_counter = util.InfectiousContactCount(
    contacts=contacts_all,
    samples=None,
    num_users=num_users,
    num_time_steps=num_time_steps,
  )

  q_marginal_infected = np.array([
    [.8, .8, .8, .8, .8],
    [.1, .1, .8, .8, .8],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
  ])

  user = 1
  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=.3,
    past_contacts=infect_counter.get_past_contacts_slice([user])[0],
    num_time_steps=num_time_steps)

  # Contact with user 0, which is infected with .8, so penalty for term should
  # be less (higher number) than no_term.
  assert np.all(d_term >= d_no_term)
  assert d_term[0] == 0
  assert d_no_term[0] == 0
  assert d_term[num_time_steps] == 0

  # Second test case
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1],
    [.1, .1, .8, .8, .8],
    [.1, .1, .1, .1, .1],
    [.1, .1, .1, .1, .1],
    [.8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1],
  ])

  user = 1
  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=1E-5,
    past_contacts=infect_counter.get_past_contacts_slice([user])[0],
    num_time_steps=num_time_steps)

  # With small p1, penalty for termination should be small (low number)
  assert np.all(d_term < 0.001)

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term_new, d_no_term_new = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.01,
    p1=1E-5,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)
  np.testing.assert_array_almost_equal(d_term, d_term_new)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_new)


def test_d_penalty_term_numerical():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 2, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ])

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term, d_no_term = util.precompute_d_penalty_terms_fn(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  d_term_new, d_no_term_new = util.precompute_d_penalty_terms_fn2(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  np.testing.assert_array_almost_equal(d_term, d_term_new)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_new)
  assert d_no_term_new.dtype == np.float32
  assert d_term_new.dtype == np.float32


def test_d_penalty_term_regression():
  contacts_all = np.array([
    (4, 1, 1, 1),
    (3, 1, 4, 1),
    ], dtype=np.int32)
  num_users = 6
  num_time_steps = 8

  # Second test case
  user = 1
  q_marginal_infected = np.array([
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.1, .1, .1, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
    [.9, .9, .9, .9, .9, .9, .9, .9],
    [.8, .8, .8, .8, .8, .8, .8, .8],
    [.1, .1, .1, .1, .1, .1, .1, .1],
  ])

  past_contacts, _ = util.get_past_contacts_static(
    (0, num_users), contacts_all, num_msg=int(num_time_steps*100))

  d_term, d_no_term = util.precompute_d_penalty_terms_fn2(
    q_marginal_infected,
    p0=0.001,
    p1=0.3,
    past_contacts=past_contacts[user],
    num_time_steps=num_time_steps)

  assert d_no_term.dtype == np.float32
  assert d_term.dtype == np.float32

  # Note: these are the results from the old implementation
  # Dump made on January 27, 2023

  d_term_expected = np.array(
    [0., 0., 5.4838, 0., 0., 5.601122, 0., 0., 0.], dtype=np.float32)
  d_no_term_expected = np.array(
    [0., 0., -0.274437, 0., 0., -0.314711, 0., 0., 0.], dtype=np.float32)

  np.testing.assert_array_almost_equal(d_term, d_term_expected)
  np.testing.assert_array_almost_equal(d_no_term, d_no_term_expected)


def test_softmax():
  # Check that trick for numerical stability yields identical results

  def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

  logits = np.random.randn(13)

  np.testing.assert_array_almost_equal(
    softmax(logits),
    util.softmax(logits)
  )


def test_it_num_infected_probs():
  # For max entropy, all probs_total should be 2**-len(probs)
  probs = [.5, .5, .5, .5]
  for _, prob_total in util.it_num_infected_probs(probs):
    np.testing.assert_almost_equal(prob_total, 1/16)

  # For any (random) probs, the probs_total should sum to 1
  probs = list((random.random() for _ in range(5)))
  sums, probs_total = zip(*util.it_num_infected_probs(probs))
  assert len(list(sums)) == 2**5
  np.testing.assert_almost_equal(sum(probs_total), 1.0)

  # For one prob, should reduce to [(0, 1-p), (1, p)]
  prob = random.random()
  sums, probs_total = zip(*util.it_num_infected_probs([prob]))
  assert list(sums) == [0, 1]
  np.testing.assert_almost_equal(list(probs_total), [1-prob, prob])

  # Manual result
  probs = [.8, .7]
  expected = [(0, .2*.3), (1, .2*.7), (1, .8*.3), (2, .8*.7)]
  expected_sums, expected_probs_total = zip(*expected)
  result_sums, result_probs_total = zip(*util.it_num_infected_probs(probs))
  np.testing.assert_array_almost_equal(list(result_sums), list(expected_sums))
  np.testing.assert_array_almost_equal(
    list(result_probs_total), list(expected_probs_total))


def test_past_contact_array():
  contacts_all = [
    (1, 2, 4, 1),
    (2, 1, 4, 1),
    (0, 1, 4, 1)
    ]

  counter = util.InfectiousContactCount(
    contacts_all, None, num_users=6, num_time_steps=7)
  past_contacts = counter.get_past_contacts_slice([0, 1, 2])

  np.testing.assert_array_almost_equal(past_contacts.shape, [3, 2+1, 3])
  np.testing.assert_array_almost_equal(past_contacts[0], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][1], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][0], [4, 1, 1])

  np.testing.assert_equal(past_contacts.dtype, np.int32)


def test_past_contact_array_fast():
  contacts_all = np.array([
    (1, 2, 4, 1),
    (2, 1, 4, 1),
    (0, 1, 4, 1)
    ])

  past_contacts, max_num_c = util.get_past_contacts_fast((0, 3), contacts_all)

  np.testing.assert_array_almost_equal(past_contacts.shape, [3, 2+1, 3])
  np.testing.assert_array_almost_equal(past_contacts[0], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][1], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][0], [4, 1, 1])

  np.testing.assert_equal(past_contacts.dtype, np.int32)
  np.testing.assert_almost_equal(max_num_c, 2)


def test_past_contact_array_static():
  num_msg = 13
  contacts_all = np.array([
    (1, 2, 4, 1),
    (2, 1, 4, 1),
    (0, 1, 4, 1)
    ], dtype=np.int32)

  past_contacts, max_num_c = util.get_past_contacts_static(
    (0, 3), contacts_all, num_msg=num_msg)

  np.testing.assert_array_almost_equal(past_contacts.shape, [3, num_msg, 3])
  np.testing.assert_array_almost_equal(past_contacts[0], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][1], -1)
  np.testing.assert_array_almost_equal(past_contacts[2][0], [4, 1, 1])

  np.testing.assert_equal(past_contacts.dtype, np.int32)
  np.testing.assert_almost_equal(max_num_c, 2)


def test_past_contact_array_fast_copy_paste():
  contacts_all = np.array([
    (1, 2, 4, 1),
    (1, 2, 3, 1),
    (1, 2, 2, 1),
    (1, 2, 1, 1),
    (2, 1, 4, 1),
    ])

  counter = util.InfectiousContactCount(
    contacts_all, None, num_users=6, num_time_steps=7)
  past_contacts_slow = counter.get_past_contacts_slice([0, 1, 2])

  past_contacts_fast, max_num_c = util.get_past_contacts_fast(
    (0, 3), contacts_all)
  np.testing.assert_almost_equal(max_num_c, 4)

  # Check that the shapes match
  np.testing.assert_array_almost_equal(
    past_contacts_slow.shape, past_contacts_fast.shape
  )

  # Check that the values match
  np.testing.assert_array_almost_equal(
    past_contacts_slow[0], past_contacts_fast[0]
  )
  np.testing.assert_array_almost_equal(
    past_contacts_slow[1], past_contacts_fast[1]
  )
  np.testing.assert_array_almost_equal(
    # Silly to test mean, but contacts could occur in any order ofcourse
    np.sum(past_contacts_slow[2]), np.sum(past_contacts_fast[2])
  )


def test_past_contact_array_fast_copy_paste_static():
  num_msg = 13
  contacts_all = np.array([
    (1, 2, 4, 1),
    (1, 2, 3, 1),
    (1, 2, 2, 1),
    (1, 2, 1, 1),
    (2, 1, 4, 1),
    ])

  past_contacts_static, max_num_static = util.get_past_contacts_static(
    (0, 3), contacts_all, num_msg=num_msg)

  past_contacts_fast, max_num_c = util.get_past_contacts_fast(
    (0, 3), contacts_all)
  np.testing.assert_almost_equal(max_num_c, 4)

  # Check that the max_contacts match
  np.testing.assert_almost_equal(
    max_num_static, max_num_c)

  # Silly to test set, but contacts could occur in any order ofcourse
  # Check that the values match
  np.testing.assert_equal(
    set(past_contacts_static[0].flatten().tolist()),
    set(past_contacts_fast[0].flatten().tolist())
  )
  np.testing.assert_equal(
    set(past_contacts_static[1].flatten().tolist()),
    set(past_contacts_fast[1].flatten().tolist())
  )
  np.testing.assert_equal(
    set(past_contacts_static[2].flatten().tolist()),
    set(past_contacts_fast[2].flatten().tolist())
  )


def test_enumerate_start_belief():
  seq_array = np.stack(list(
    util.iter_sequences(time_total=5, start_se=False)))

  start_belief = np.array([.1, .2, .3, .4])
  A_start_belief = util.enumerate_start_belief(seq_array, start_belief)

  expected = np.array(
    [0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
     0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

  np.testing.assert_array_almost_equal(A_start_belief, expected)


def test_spread_buckets():
  num_sample_array = util.spread_buckets(100, 10)
  expected = 10 * np.ones((10))
  np.testing.assert_array_almost_equal(num_sample_array, expected)

  num_sample_array = util.spread_buckets(97, 97)
  np.testing.assert_array_almost_equal(num_sample_array, np.ones((97)))

  num_samples = np.sum(util.spread_buckets(100, 13))
  np.testing.assert_almost_equal(num_samples, 100)

  with np.testing.assert_raises(AssertionError):
    util.spread_buckets(13, 17)


def test_spread_buckets_interval():
  user_id = util.spread_buckets_interval(100, 10)
  np.testing.assert_array_almost_equal(user_id, 10*np.arange(11))


def test_quantize():
  x = np.random.randn(13, 13).astype(np.float32)

  x_quantized = util.quantize(x, 8)
  assert x_quantized.dtype == np.float32

  x_quantized = util.quantize(x, -1)
  assert x_quantized.dtype == np.float32


def test_get_stale_users_binary():
  stales = np.array([7, 9, 13], dtype=np.int32)

  num_users = 18
  result = util.get_stale_users_binary(stales, num_users=num_users)

  np.testing.assert_array_almost_equal(result.shape, [num_users, 1])
  assert result.dtype == np.bool

  np.testing.assert_almost_equal(result[6], 1)
  np.testing.assert_almost_equal(result[7], 0)
  np.testing.assert_almost_equal(result[8], 1)
