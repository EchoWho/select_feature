import numpy as np
import grain_common

def merge_loss(L1, L2, err1, err2, w1, w2):
  L = {}
  err = {}
  w = np.float64(w1 + w2)
  w1 /= w
  w2 /= w
  for method in L1.keys():
    L[method] = w1 * L1[method] + w2 * L2[method]
    err[method] = w1 * err1[method] + w2 * err2[method]
  return L, err, w


def load_loss(filename):
  d = np.load(filename)
  L = d['L']
  err = d['err']
  L = L.item()
  err = err.item()
  w = np.float64(d['filesize'])
  costs = d['costs']
  costs = costs.item()
  return L, err, w, costs


if __name__ == "__main__":
  filenames = [ ]
  for i in range(5):
    filenames.append(grain_common.filename_budget_vs_loss(i+1))
    print filenames[-1]

  is_first = True
  for _, filename in enumerate(filenames):
    L2, err2, w2, costs2 = load_loss(filename)
    if is_first:
      is_first = False
      L = L2
      err = err2
      w = w2
      costs = costs2
    else:
      L, err, w = merge_loss(L, L2, err, err2, w, w2)
  filename = grain_common.filename_budget_vs_loss(0)
  np.savez(filename, L=L, err=err, costs=costs, filesize=w)
