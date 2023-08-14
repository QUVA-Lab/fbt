"""Compare inference methods on likelihood and AUROC, and run prequentially."""
import argparse
import copy
import numpy as np
from fbt.config import config
from fbt.data import data_load
from fbt.experiments import prequential, util_experiments
from fbt import constants
from fbt import LOGGER_FILENAME, logger
from fbt import simulator
from fbt import util
import numba
import os
import psutil
import random
from sklearn import metrics
import time
import tqdm
import traceback
from typing import Any, Dict, List, Optional
import wandb
import csv

import pandas as pd
import networkx as nx


def make_inference_func(
    inference_method: str,
    num_users: int,
    cfg: Dict[str, Any],
    ):
  """Pulls together the inference function with parameters.

  Args:
    inference_method: string describing the inference method
    num_users: number of users in this simulation
    cfg: the configuration dict generated upon init of the experiment

  Returns:
    the inference function (input: data; output: marginals over SEIR per user)
  """
  p0 = cfg["model"]["p0"]
  p1 = cfg["model"]["p1"]
  g = cfg["model"]["prob_g"]
  h = cfg["model"]["prob_h"]
  alpha = cfg["model"]["alpha"]
  beta = cfg["model"]["beta"]
  quantization = cfg["model"]["quantization"]

  # Construct dynamics
  # Construct Geometric distro's for E and I states

  do_random_quarantine = False
  if inference_method == "fn":
    inference_func = util_experiments.wrap_fact_neigh_inference(
      num_users=num_users,
      alpha=alpha,
      beta=beta,
      p0=p0,
      p1=p1,
      g_param=g,
      h_param=h,
      quantization=quantization)
  elif inference_method == "random":
    inference_func = None
    do_random_quarantine = True
  elif inference_method == "dummy":
    inference_func = util_experiments.wrap_dummy_inference(
      num_users=num_users)
  else:
    raise ValueError((
      f"Not recognised inference method {inference_method}. Should be one of"
      f"['random', 'fn', 'dummy']"
    ))
  return inference_func, do_random_quarantine


def compare_prequential_quarantine(
    inference_method: str,
    num_users: int,
    num_time_steps: int,
    observations: List[constants.Observation],
    contacts: List[constants.Contact],
    states: np.ndarray,
    cfg: Dict[str, Any],
    runner,
    results_dir: str,
    quick: bool = False,
    do_diagnosis: bool = False,
    use_abm_simulator: bool = False):
  """Compares different inference algorithms on the supplied contact graph."""
  del states

  num_users = int(num_users)
  del observations

  num_days_window = cfg["model"]["num_days_window"]
  quantization = cfg["model"]["quantization"]

  num_rounds = cfg["model"]["num_rounds"]
  rng_seed = cfg.get("sim_seed", 123)

  fraction_test = cfg["data"]["fraction_test"]
  positive_e_state = bool(cfg["data"]["positive_e_state"])

  policy = cfg["model"]["policy"]
  test_stochastic = cfg["model"]["test_stochastic"]
  hybrid_split = cfg["model"]["hybrid_split"]
  oracle = int(cfg["model"]["oracle"])
  num_time_steps = cfg["model"]["t_days"]
  feature_weight_0 = cfg["model"]["feature_weight_0"]
  feature_weight_1 = cfg["model"]["feature_weight_1"]
  feature_weight_2 = cfg["model"]["feature_weight_2"]
  feature_pow_1 = cfg["model"]["feature_pow_1"]
  feature_pow_2 = cfg["model"]["feature_pow_2"]

  # Data and simulator params
  fraction_stale = cfg["data"]["fraction_stale"]
  num_days_quarantine = cfg["data"]["num_days_quarantine"]
  t_start_quarantine = cfg["data"]["t_start_quarantine"]

  probab_0 = cfg["model"]["p0"]
  params_dynamics = {
    "p0": cfg["data"]["p0"],
    "p1": cfg["data"]["p1"],
    "g": cfg["data"]["prob_g"],
    "h": cfg["data"]["prob_h"],
  }

  logger.info((
    f"Settings at experiment: {quantization:.0f} quant, at {fraction_test}%"))

  users_stale = None
  if fraction_stale > 0:
    users_stale = np.random.choice(
      num_users, replace=False, size=int(fraction_stale*num_users))

  diagnostic = runner if do_diagnosis else None

  inference_func, do_random_quarantine = make_inference_func(
    inference_method, num_users, cfg)

  # Set conditional distributions for observations
  p_obs_infected = np.array(
    [cfg["data"]["alpha"], 1-float(cfg["data"]["alpha"])], dtype=np.float32)
  p_obs_not_infected = np.array(
    [1-float(cfg["data"]["beta"]), cfg["data"]["beta"]], dtype=np.float32)

  start_belief_global = (
    np.ones((num_users, 4)) * np.array([1. - probab_0, probab_0, 0., 0.]))

  if quick:
    num_rounds = 2

  # Arrays to accumulate statistics
  pir = 0.
  phr = 0.
  time_pir = 0.
  time_phr = 0.
  phr_ages = []
  pir_ages = []
  precisions = np.zeros((num_time_steps))
  recalls = np.zeros((num_time_steps))
  infection_rates = np.zeros((num_time_steps))
  hospital_rates = np.zeros((num_time_steps))
  exposed_rates = np.zeros((num_time_steps))
  likelihoods_state = np.zeros((num_time_steps))
  ave_prob_inf = np.zeros((num_time_steps))
  ave_prob_inf_at_inf = np.zeros((num_time_steps))
  ave_precision = np.zeros((num_time_steps))
  num_quarantined = np.zeros((num_time_steps), dtype=np.int32)
  num_tested = np.zeros((num_time_steps), dtype=np.int32)

  # Placeholder for tests on first day
  z_states_inferred = np.zeros((num_users, 1, 4))
  user_quarantine_ends = -1*np.ones((num_users), dtype=np.int32)

  logger.info(f"Do random quarantine? {do_random_quarantine}")
  t0 = time.time()

  if use_abm_simulator:
    sim_factory = simulator.ABMSimulator
    contacts = []
  else:
    sim_factory = simulator.CRISPSimulator

  sim = sim_factory(
    num_time_steps, num_users, params_dynamics, positive_e_state, rng_seed)
  sim.init_day0(copy.deepcopy(contacts))

  logger.info((
    f"Start simulation with {num_rounds} updates"))

  # Get the household size of every house in the simulation
  household_info = pd.read_csv(
    '../abm/tests/data/baseline_household_demographics.csv').sum(
      axis=1).tolist()
  pop_info = sim.get_pop_info()
  pop_info['individual_id'] = range(num_users)
  pop_info["house_size"] = pop_info.apply(
    lambda row: household_info[row.house], axis=1)
  degree_hist = [[0] for _ in range(num_users)]
  age_pirs = {}
  house_pirs = {}

  all_score_history = []
  score_history = []
  state_history = []

  for t_now in tqdm.trange(1, num_time_steps):

    t_start_loop = time.time()

    sim.step()

    # Number of days to use for inference
    num_days = min((t_now + 1, num_days_window))
    # When num_days exceeds t_now, then offset should start counting at 0
    days_offset = t_now + 1 - num_days
    assert 0 <= days_offset <= num_time_steps

    # For each day, t_now, only receive obs up to and including 't_now-1'
    assert sim.get_current_day() == t_now

    rank_score = (z_states_inferred[:, -1, 1] + z_states_inferred[:, -1, 2])
    all_score_history.append(rank_score)

    # Do not test when user in quarantine
    rank_score *= (user_quarantine_ends < t_now)
    score_history.append(rank_score)
    state_history.append(sim.get_states_abm())

    # Oracle policy that tests people based on their true state, not inference
    if oracle in (2, 3):
      # Create state 0 for individuals currently in quarantine
      current_states = (sim.get_states_today()+1)
      current_states *= (user_quarantine_ends < t_now)
      users_to_test = np.array([], dtype=np.int32)
      n_test = int(fraction_test * num_users)

      # Adjusts the priority of the Exposed state (number 2)
      if oracle == 2:
        nstates = [3, 1, 4, 0, 2]
      else:
        nstates = [3, 2, 1, 4, 0]

      for nstate in nstates:
        # Pick everyone in this state if possible, else randomly sample
        state_indices = np.argwhere(current_states == nstate).flatten()
        users_to_test = np.append(users_to_test,
          np.random.choice(a=state_indices,
                            size=min(len(state_indices), n_test),
                            replace=False))
        n_test -= len(state_indices)
        if n_test <= 0:
          break
    else:
      # Grab tests on the main process
      users_to_test = prequential.decide_tests(
        scores_infect=rank_score,
        num_tests=int(fraction_test * num_users),
        rand_gen=arg_rng,
        pop_info=pop_info,
        policy=policy,
        stochastic=test_stochastic,
        hybrid_split=hybrid_split,
        feature_weight_0=feature_weight_0,
        feature_weight_1=feature_weight_1,
        feature_weight_2=feature_weight_2,
        feature_pow_1=feature_pow_1,
        feature_pow_2=feature_pow_2,
        degrees=[user_hist[-1] for user_hist in degree_hist])

    obs_today = sim.get_observations_today(
      users_to_test.astype(np.int32),
      p_obs_infected,
      p_obs_not_infected,
      arg_rng)

    sim.set_window(days_offset)

    if not do_random_quarantine:
      t_start = time.time()

      # Make inference over SEIR states
      start_belief = start_belief_global
      if num_days <= t_now:
        logger.info("Use window!")
        start_belief = z_states_inferred[:, 1]

      contacts_now = util.make_default_array(
        sim.get_contacts(), dtype=np.int32, rowlength=4)
      observations_now = util.make_default_array(
        sim.get_observations_all(), dtype=np.int32, rowlength=3)

      num_contacts = np.array(contacts_now.shape[0], dtype=np.int32)
      num_obs = np.array(observations_now.shape[0], dtype=np.int32)

      logger.info(f"Day {t_now}: {num_contacts} contacts, {num_obs} obs")

      z_states_inferred = inference_func(
        observations_now,
        contacts_now,
        num_rounds,
        num_days,
        start_belief,
        users_stale=users_stale,
        diagnostic=diagnostic)

      np.testing.assert_array_almost_equal(
        z_states_inferred.shape, [num_users, num_days, 4])
      logger.info(f"Time spent on inference_func {time.time() - t_start:.0f}")

      # Decide who to quarantine and subtract contacts
      if oracle == 1:
        # Oracle policy that directly quarantines infected and exposed people
        current_states = sim.get_states_today()
        users_to_quarantine = np.array([], dtype=np.int32)

        for nstate in [2, 1]:
          state_indices = np.argwhere(current_states == nstate).flatten()
          users_to_quarantine = np.append(users_to_quarantine, state_indices)

      elif oracle == 4:
        # Age based oracle policy that directly quarantines elderly population
        ages = pop_info["age"].to_numpy()
        users_to_quarantine = (ages > 6).nonzero()[0]

    else:
      z_states_inferred = np.zeros((num_users, num_days, 4))
      users_to_quarantine = np.random.choice(
        num_users, size=int(0.05*num_users)).tolist()

    if oracle not in (1, 4):
      logger.info("Conditional quarantine")
      users_to_quarantine = obs_today[np.where(obs_today[:, 2] > 0)[0], 0]

    if t_now < t_start_quarantine:
      users_to_quarantine = np.array([], dtype=np.int32)

    user_quarantine_ends[users_to_quarantine] = t_now + num_days_quarantine

    # This function will remove the contacts that happen TODAY (and which may
    # spread the virus and cause people to shift to E-state tomorrow).
    sim.quarantine_users(users_to_quarantine, num_days_quarantine)
    assert sim.get_current_day() == t_now

    # NOTE: fpr is only defined as long as num_users_quarantine is fixed.
    # else switch to precision and recall
    states_today = sim.get_states_today()

    precision, recall = prequential.calc_prec_recall(
      states_today, users_to_quarantine)

    infected = (states_today == 2)
    infection_rate = np.mean(infected)
    hospitalized = ((state_history[-1] >= 6) & (state_history[-1] <= 8))
    hospital_rate = np.mean(hospitalized)

    exposed_rate = np.mean(
      np.logical_or(states_today == 1, states_today == 2))

    if infection_rate > pir:
      pir = infection_rate
      time_pir = t_now
      pir_ages = [np.mean(
        (pop_info["age"].to_numpy() == i)*infected) for i in range(9)]

    if hospital_rate > phr:
      phr = hospital_rate
      time_phr = t_now
      phr_ages = [np.mean(
        (pop_info["age"].to_numpy() == i)*hospitalized) for i in range(9)]

    logger.info((f"precision: {precision:5.2f}, recall: {recall: 5.2f}, "
                  f"infection rate: {infection_rate:5.3f}({pir:5.3f}),"
                  f"hospital rate: {hospital_rate:5.3f}({phr:5.3f}),"
                  f"{exposed_rate:5.3f}, tests: {len(users_to_test):5.0f} "
                  f"Qs: {len(users_to_quarantine):5.0f}"))

    precisions[t_now] = precision
    recalls[t_now] = recall
    infection_rates[t_now] = infection_rate
    hospital_rates[t_now] = hospital_rate
    exposed_rates[t_now] = np.mean(
      np.logical_or(states_today == 1, states_today == 2))
    num_quarantined[t_now] = len(users_to_quarantine)
    num_tested[t_now] = len(users_to_test)

    # Inspect using sampled states
    p_at_state = z_states_inferred[range(num_users), num_days-1, states_today]
    likelihoods_state[t_now] = np.mean(np.log(p_at_state + 1E-9))
    ave_prob_inf_at_inf[t_now] = np.mean(
      p_at_state[states_today == 2])
    ave_prob_inf[t_now] = np.mean(z_states_inferred[:, num_days-1, 2])

    if infection_rate > 0:
      ave_precision[t_now] = metrics.average_precision_score(
        y_true=(states_today == 2),
        y_score=z_states_inferred[:, num_days-1, 2])
    else:
      ave_precision[t_now] = 0.

    time_full_loop = time.time() - t_start_loop
    logger.info(f"Time spent on full_loop {time_full_loop:.0f}")

    # Build network of daily contacts
    contacts_day = sim.get_contacts_daily()
    contacts_graph = nx.DiGraph([(edge[0], edge[1]) for edge in contacts_day])
    norm_degrees = nx.degree_centrality(contacts_graph)

    # Calculate un-normalized degrees and add to degree history
    n_nodes = contacts_graph.number_of_nodes()
    degrees = {key: value*(n_nodes-1) for key, value in norm_degrees.items()}
    for user in range(num_users):
      degree_hist[user].append(degrees.get(user, 0))

    if t_now == 1:
      pop_info["init_deg"] = [
        degree_hist[user][0] for user in range(num_users)]

    pop_info["state"] = sim.get_states_today()

    for age in pop_info['age'].unique():
      age_pirs[f"age_{age}"] = max(
        np.mean(pop_info[pop_info['age'] == age]['state'] == 2),
        age_pirs.get(f"age_{age}", 0))

    for house in pop_info['house_size'].unique():
      house_pirs[f"house_{house}"] = max(
        np.mean(pop_info[pop_info['house_size'] == house]['state'] == 2),
        house_pirs.get(f"house_{house}", 0))

    loadavg1, loadavg5, _ = os.getloadavg()
    swap_use = psutil.swap_memory().used / (1024.0 ** 3)

    runner.log({**{
      "time_step": time_full_loop,
      "infection_rate": infection_rate,
      "hospital_rate": hospital_rate,
      "load1": loadavg1,
      "load5": loadavg5,
      "swap_use": swap_use,
      "recall": recall,
      "max_score": max(rank_score),
      }})


  logger.info(f"At day {time_pir} peak infection rate is {pir}")

  prequential.dump_results_json(
    datadir=results_dir,
    cfg=cfg,
    ave_prob_inf=ave_prob_inf.tolist(),
    ave_prob_inf_at_inf=ave_prob_inf_at_inf.tolist(),
    ave_precision=ave_precision.tolist(),
    exposed_rates=exposed_rates.tolist(),
    inference_method=inference_method,
    infection_rates=infection_rates.tolist(),
    hospital_rates=hospital_rates.tolist(),
    likelihoods_state=likelihoods_state.tolist(),
    name=runner.name,
    num_quarantined=num_quarantined.tolist(),
    num_tested=num_tested.tolist(),
    pir=float(pir),
    precisions=precisions.tolist(),
    quantization=quantization,
    recalls=recalls.tolist(),
    seed=cfg.get("model_seed", -1),
  )

  time_spent = time.time() - t0
  logger.info(f"With {num_rounds} rounds, PIR {pir:5.2f}")
  runner.log({**{
    "time_spent": time_spent,
    "pir": pir,
    "pir_time": time_pir,
    "phr": phr,
    "phr_time": time_phr,
    "tot_hosp": np.mean([(6 in user) for user in np.array(state_history).T]),
    "n_suscep": np.count_nonzero(states_today == 0),},
    **{f"pir_age{i}": pir_age for i, pir_age in enumerate(pir_ages)},
    **{f"phr_age{i}": phr_age for i, phr_age in enumerate(phr_ages)}})
    # **age_pirs, **house_pirs,


  if do_diagnosis:
    # Write data such as individual attributes to files
    sim.sim.env.model.write_output_files()

    with open(
      "results/prequential/initial_degrees.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerows(pop_info["init_deg"])
    with open(
        "results/prequential/score_history.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerows(score_history)
    with open(
        "results/prequential/all_score_history.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerows(all_score_history)
    with open(
        "results/prequential/cent_history.csv", "w", newline="") as f:
      writer = csv.writer(f)
      sort_hist = [value for key,value in sorted(degree_hist.items())]
      sort_hist = [list(i) for i in zip(*sort_hist)]
      writer.writerows(sort_hist)


  # Overwrite every experiment, such that code could be pre-empted
  prequential.dump_results(
    results_dir, precisions=precisions, recalls=recalls,
    infection_rates=infection_rates, hospital_rates=hospital_rates)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compare statistics acrosss inference methods')
  parser.add_argument('--inference_method', type=str, default='fn',
                      choices=[
                        'fn', 'dummy', 'random'],
                      help='Name of the inference method')
  parser.add_argument('--experiment_setup', type=str, default='prequential',
                      help='Name of the experiment_setup')
  parser.add_argument('--config_data', type=str, default='large_graph_02',
                      help='Name of the config file for the data')
  parser.add_argument('--config_model', type=str, default='model_02',
                      help='Name of the config file for the model')
  parser.add_argument('--name', type=str, default=None,
                      help=('Name of the experiments. WandB will set a random'
                            ' when left undefined'))
  parser.add_argument('--do_diagnosis', action='store_true')
  parser.add_argument('--quick', action='store_true',
                      help=('include flag --quick to run a minimal version of'
                            'the code quickly, usually for debugging purpose'))

  numba.set_num_threads(util.get_cpu_count())

  args = parser.parse_args()

  configname_data = args.config_data
  configname_model = args.config_model
  fname_config_data = f"fbt/config/{configname_data}.ini"
  fname_config_model = f"fbt/config/{configname_model}.ini"
  data_dir = f"fbt/data/{configname_data}/"
  do_abm = '_abm' in configname_data

  inf_method = args.inference_method
  # Set up locations to store results
  experiment_name = args.experiment_setup
  if args.quick:
    experiment_name += "_quick"
  results_dir_global = (
    f'results/{experiment_name}/{configname_data}__{configname_model}/')

  util.maybe_make_dir(results_dir_global)

  config_data = config.ConfigBase(fname_config_data)
  config_model = config.ConfigBase(fname_config_model)

  # Start WandB
  config_wandb = {
    "config_data_name": configname_data,
    "config_model_name": configname_model,
    "cpu_count": util.get_cpu_count(),
    "data": config_data.to_dict(),
    "model": config_model.to_dict(),
  }

  # WandB tags
  tags = [
    args.experiment_setup, inf_method, f"cpu{util.get_cpu_count()}"]
  tags.append("quick" if args.quick else "noquick")
  tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")

  runner_global = wandb.init(
    project="fbt",
    notes=" ",
    name=args.name,
    tags=tags,
    config=config_wandb,
  )

  config_wandb = config.clean_hierarchy(dict(runner_global.config))
  config_wandb = util_experiments.set_noisy_test_params(config_wandb)
  logger.info(config_wandb)

  logger.info(f"Logger filename {LOGGER_FILENAME}")
  logger.info(f"Saving to results_dir_global {results_dir_global}")
  logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
  logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
  logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
  logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

  util_experiments.make_git_log()

  # Set random seed
  seed_value = config_wandb.get("model_seed", None)
  random.seed(seed_value)
  np.random.seed(seed_value)
  # Random number generator to pass as argument to some imported functions
  arg_rng = np.random.default_rng(seed=seed_value)

  if not do_abm:
    if not os.path.exists(data_dir):
      raise FileNotFoundError((
        f"{data_dir} not found. Current wd: {os.getcwd()}"))

    observations_all, contacts_all, states_all = data_load.load_jsons(data_dir)
  else:
    observations_all = contacts_all = []
    states_all = np.zeros((1, 1, 1))

  try:
    compare_prequential_quarantine(
      inf_method,
      num_users=config_wandb["data"]["num_users"],
      num_time_steps=config_wandb["data"]["num_time_steps"],
      observations=observations_all,
      contacts=contacts_all,
      states=states_all,
      cfg=config_wandb,
      runner=runner_global,
      results_dir=results_dir_global,
      quick=args.quick,
      do_diagnosis=args.do_diagnosis,
      use_abm_simulator=do_abm,
      )
  except Exception as e:
    # This exception sends an WandB alert with the traceback and sweepid
    logger.info(f'Error repr: {repr(e)}')
    traceback_report = traceback.format_exc()
    wandb.alert(
      title=f"Error {os.getenv('SWEEPID')}-{os.getenv('SLURM_JOB_ID')}",
      text=(
        f"'{configname_data}', '{configname_model}', '{inf_method}'\n"
        + traceback_report)
    )
    raise e

  runner_global.finish()
