import matplotlib.pyplot as plt
import numpy as np
import yahoo_common


d = np.load('yahoo_results/budget_vs_loss.group.0.npz')
L = d['L']
b = d['budget']
print L

d_nw = np.load('yahoo_results/budget_vs_loss.group.0.no_whiten.npz')
L_nw = d_nw['L']
L_nw = np.minimum(2, L_nw)
b_nw = d_nw['budget']

show_both=False

if show_both:
  plt.plot(b, L[0], 'r', b_nw, L_nw[0], 'g', b_nw, L_nw[1], 'b') 
elif yahoo_common.whiten:
  plt.plot(b, L[0], 'r', b, L[2], 'b')
else:
  plt.plot(b_nw, L_nw[0], 'r', b_nw, L_nw[1], 'g', b_nw, L_nw[2], 'b')

plt.show()
