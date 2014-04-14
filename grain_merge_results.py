import numpy as np

def merge_loss(L1, L2, w1, w2):
  L = {}
  w = np.float64(w1 + w2)
  w1 /= w
  w2 /= w
  for method in L1.keys():
    L[method] = w1 * L1[method] + w2 * L2[method]
  return L, w


def load_loss(filename):
  d = np.load(filename)
  L = d['L']
  L = L.item()
  w = np.float64(d['filesize'])
  costs = d['costs']
  costs = costs.item()
  return L, w, costs


if __name__ == "__main__":
  filenames = [ ]
  is_first = True
  for _, filename in enumerate(filenames):
    L2, w2, costs2 = load_loss(filename)
    if is_first:
      is_first = False
      L = L2
      w = w2
      cost = cost2
    else:
      L, w = merge_loss(L, L2, w, w2)
  np.savez(filename, L=L, costs=costs, filesize=w)
