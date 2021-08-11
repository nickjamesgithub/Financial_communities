from sortedcontainers import SortedList
import numpy as np
from math import log

def log_grad(x1, y1, x2, y2):
    if y2 <= 0:
        return float('-inf')
    if y1 <= 0 or x1 == x2:
        return float('inf')
    return (log(y2) - log(y1)) / (x2 - x1)


def element_to_set_distance(v, s):
    """Returns shortest distance of some value to any element of a set."""
    return min([abs(v - x) for x in s])


def set_to_set_distance(s1, s2):
    """Returns distance between two sets, as defined in original paper."""
    if len(s1) == 0 or len(s2) == 0:
        return float('inf')
    return 0.5 * (sum([element_to_set_distance(e, s2) for e in s1]) / len(s1) +
                  sum([element_to_set_distance(e, s1) for e in s2]) / len(s2))


def turning_point_set_distance(tps1, tps2):
    """
    Returns distance between two sets of turning points. This is defined as the
    distance between the two sets of peaks plus the distance between the two
    sets of troughs.
    """
    # Assume both sets alternate T/P and start with T, P, ...
    if len(tps1) < 2 or len(tps2) < 2:
        return float('inf')
    return (set_to_set_distance(tps1[0::2], tps2[0::2]) +
            set_to_set_distance(tps1[1::2], tps2[1::2]))

def identify_turning_points(
    x_raw, local_radius=17, peak_ratio=0.2, min_log_grad=0.01, proportion_of_global_max = 0.005):
  """
  Identifies the set of 'turning points' in a time series.
  Time complexity is O(N log(local_radius)).
  Parameters
  ----------
  x_raw : array_like
      the time series, should be convertible to a 1D numpy array
  local_radius : int
      a peak/trough must satisfy the condition of being the max/min of any
      values within 'local_radius' time steps forwards or backwards
  peak_ratio : float
      a peak must satisfy the condition of being at least peak_ratio * the
      value of the previous peak
  min_log_grad : float
      a turning point must satisfy the condition of having a log_gradient
      magnitude of at least min_log_grad from the previous turning point
  Returns
  -------
  array-like
    sequence of 0-based indices representing the identified turning points.
    The first turning point will be a trough, and proceed to alternate between
    peak and trough.
  """

  x = np.array(x_raw)
  x[x<0] = 0
  M = np.max(x)

  # Preprocess: cache right-side peak/trough neighbourhood validity, O(N logN)
  # valid_peak[i] = True iff x[i] >= max(x[i+1], ..., x[i+local_radius])
  # valid_trough[i] = True iff x[i] <= min(x[i+1], ..., x[i+local_radius])
  valid_peak = np.full((len(x)), False)
  valid_trough = np.full((len(x)), False)
  next_values = SortedList([x[-1]])
  valid_peak[-1] = True
  valid_trough[-1] = True
  for i in range(len(x)-2, -1, -1):
    valid_peak[i] = x[i] >= next_values[-1]
    valid_trough[i] = x[i] <= next_values[0]
    if i + local_radius < len(x):
      next_values.remove(x[i+local_radius])  # O(log l)
    next_values.add(x[i])  # O(log l)

  # For now, we assume the first TP will be a trough.
  # TODO: Generalise to allow for starting at a peak.
  tps = [0]
  recent_values = SortedList([x[0]])
  for i in range(1, len(x)):
    # Update peak/trough validity based on left-side neighbourhood.
    valid_peak[i] &= (x[i] >= recent_values[-1])
    valid_trough[i] &= (x[i] <= recent_values[0])

    if len(tps) % 2 == 1:
      # The last TP we addded was a trough (odd number of turning points).
      if x[i] < x[tps[-1]]:
        # Replace last trough with this lower one.
        tps[-1] = i
      elif (x[i] > x[tps[-1]]
            and valid_peak[i]
            and (len(tps) < 2 or x[i] >= x[tps[-2]] * peak_ratio)
            and abs(log_grad(tps[-1], x[tps[-1]], i, x[i])) >= min_log_grad) \
              and x[i]>= proportion_of_global_max*M: # Add in proportion of global max * M

        # New peak: greater-or-equal to surrounding 'l' values and greater than
        # previous trough and passes peak ratio check with prev peak and
        # log_grad ratio check with prev trough.
        tps.append(i)
    else:
      # The last TP we added was a peak.
      if x[i] > x[tps[-1]]:
        # Replace recent peak with this one.
        tps[-1] = i
      elif (x[i] < x[tps[-1]]
            and valid_trough[i]
            and abs(log_grad(tps[-1], x[tps[-1]], i, x[i])) >= min_log_grad):
        # New trough: less-or-equal to surrounding 'l' values and less than
        # previous peak and passes log_grad ratio check with prev peak.
        tps.append(i)
    if i >= local_radius:
      recent_values.remove(x[i-local_radius])
    recent_values.add(x[i])
  return tps