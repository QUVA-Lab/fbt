"""Run only the simulator and compare computatinal requirements."""
import argparse
from COVID19 import model as abm_model
from COVID19 import simulation
import covid19
from fbt import constants
from fbt.config import config
from fbt.experiments import util_experiments
from fbt import LOGGER_FILENAME, logger
from fbt import util
import numpy as np
import os
import traceback
import wandb


def main(cfg, runner):
  """Runs only the ABM simulator."""
  num_time_steps = 100
  num_users = cfg["data"]["num_users"]
  do_collect = cfg["settings"]["do_collect"]
  days_of_interactions = cfg["settings"]["days_of_interactions"]
  filename = "baseline_parameters.csv"
  filename_hh = "baseline_household_demographics.csv"

  input_param_file = os.path.join(constants.ABM_HOME, filename)
  input_households = os.path.join(constants.ABM_HOME, filename_hh)

  util.check_exists(input_param_file)
  util.check_exists(input_households)
  util.maybe_make_dir("results/tmp/")

  logger.info("Construct ABM simulator")
  params = abm_model.Parameters(
    input_param_file=input_param_file,
    param_line_number=1,
    output_file_dir="results/tmp/",
    input_households=input_households
  )

  if num_users < 10000:
    # TODO figure out why this fails
    logger.debug('ABM simulator might fail with <10k users')

  # Start with sufficient amount of initial infections. Start in E-state
  n_seed = 1 + int(num_users // 2500)
  params.set_param("n_total", num_users)
  params.set_param("n_seed_infection", n_seed)
  params.set_param("days_of_interactions", days_of_interactions)

  model_init = abm_model.Model(params)
  model = simulation.COVID19IBM(model=model_init)
  sim = simulation.Simulation(env=model, end_time=num_time_steps)
  logger.info("Finished constructing ABM simulator")

  logger.info((
    f"Start experiment with {days_of_interactions} days of "
    f"interactions and {num_users} users"))

  for t_now in range(num_time_steps):
    sim.steps(1)
    datalog = {'time': t_now}

    if do_collect >= 1:
      contacts_incoming = covid19.get_contacts_daily(
        model.model.c_model, t_now)
      logger.info(
        f"Num contacts {do_collect} at day {t_now}: {len(contacts_incoming)}")
      # Get state
      state = np.array(covid19.get_state(model.model.c_model), dtype=np.int32)
      logger.info(f"State of shape {state.shape}")
      datalog["num_contacts_per_user"] = (
        float(len(contacts_incoming)) / num_users)

    if do_collect >= 2:
      users_to_quarantine = np.random.choice(
        int(.9*num_users), replace=False, size=int(.1*num_users))
      status = covid19.intervention_quarantine_list(
        model.model.c_model,
        list(users_to_quarantine),
        t_now + 5)
      assert status == 0
      state = np.array(covid19.get_state(model.model.c_model), dtype=np.int32)
      logger.info(f"State of shape {state.shape}")
    runner.log(datalog)

  contacts_final = list(
    covid19.get_contacts_daily(model.model.c_model, num_time_steps-1))
  logger.info(f"Num contacts at final day: {len(contacts_final)}")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Run only the ABM simulator')
  parser.add_argument('--config_data', type=str, default='large_graph_02',
                      help='Name of the config file for the data')
  parser.add_argument('--config_model', type=str, default='model_02',
                      help='Name of the config file for the model')

  args = parser.parse_args()

  configname_data = args.config_data
  configname_model = args.config_model
  fname_config_data = f"fbt/config/{configname_data}.ini"
  fname_config_model = f"fbt/config/{configname_model}.ini"

  config_data = config.ConfigBase(fname_config_data)
  config_model = config.ConfigBase(fname_config_model)

  # Start WandB
  config_wandb = {
    "config_data_name": configname_data,
    "config_model_name": configname_model,
    "cpu_count": util.get_cpu_count(),
    "data": config_data.to_dict(),
    "model": config_model.to_dict(),
    "settings": {
      "do_collect": 2,
      "days_of_interactions": 7,
    }
  }

  # WandB tags
  tags = ["Simulator_only", f"cpu{util.get_cpu_count()}"]
  tags.append("local" if (os.getenv('SLURM_JOB_ID') is None) else "slurm")

  runner_global = wandb.init(
    project="fbt",
    notes=" ",
    tags=tags,
    config=config_wandb,
  )

  config_wandb = config.clean_hierarchy(dict(runner_global.config))
  config_wandb = util_experiments.set_noisy_test_params(config_wandb)
  logger.info(config_wandb)

  logger.info(f"Logger filename {LOGGER_FILENAME}")
  logger.info(f"sweep_id: {os.getenv('SWEEPID')}")
  logger.info(f"slurm_id: {os.getenv('SLURM_JOB_ID')}")
  logger.info(f"slurm_name: {os.getenv('SLURM_JOB_NAME')}")
  logger.info(f"slurm_ntasks: {os.getenv('SLURM_NTASKS')}")

  util_experiments.make_git_log()

  try:
    main(cfg=config_wandb, runner=runner_global)

  except Exception as e:
    # This exception sends an WandB alert with the traceback and sweepid
    logger.info(f'Error repr: {repr(e)}')
    traceback_report = traceback.format_exc()
    wandb.alert(
      title=f"Error {os.getenv('SWEEPID')}-{os.getenv('SLURM_JOB_ID')}",
      text=(traceback_report)
    )
    raise e

  runner_global.finish()
