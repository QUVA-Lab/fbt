"""Utility functions for plots in the paper."""
import matplotlib.pyplot as plt
import os


COLOR_MAP_PARETO = {
  "Random": "c",
  "FN": 'm',
  'no_quarant': 'g',
}


SEIR_COLORS = ['b', 'm', 'r', 'k']
SEIR_STATES = ['S', 'E', 'I', 'R']

SEIR_STATE_COLOR = dict(zip(SEIR_STATES, SEIR_COLORS))


def pareto_experiment_to_color(name: str) -> str:
  """Returns the color corresponding to an experiment name."""
  for key, value in COLOR_MAP_PARETO.items():
    if key.lower() in name.lower():
      return value
  return "k"


def save_figure(dirname: str, name: str, extension: str, dpi: int = 500):
  """Saves a pyplot figure to disk."""
  fname = os.path.join(dirname, f"{name}.{extension}")
  plt.savefig(fname, dpi=dpi, bbox_inches='tight')
