"""Load the data."""
import os
import json
from fbt import constants, logger, util
import numpy as np

from typing import List, Tuple


def load_jsons(
    data_dir: str) -> Tuple[
      List[constants.Observation], List[constants.Contact], np.ndarray]:
  """Loads JSONs corresponding to the data directory."""
  fname_obs = os.path.join(data_dir, "observations.json")
  fname_contacts = os.path.join(data_dir, "contacts.json")
  fname_states = os.path.join(data_dir, "states.json")

  with open(fname_obs, 'r', encoding='UTF-8') as fp:
    observations_all = json.load(fp)

  with open(fname_contacts, 'r', encoding='UTF-8') as fp:
    contacts_all = json.load(fp)

  with open(fname_states, 'rb') as fp:
    states = np.load(fp)["states"]

  logger.info(f"Found {len(observations_all)} observations")

  # Simplify contacts and observations
  observations_all = util.make_plain_observations(observations_all)
  contacts_all = util.make_plain_contacts(contacts_all)

  return observations_all, contacts_all, states
