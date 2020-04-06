import matplotlib.pyplot as plt
import time
import seaborn as sb
import numpy as np

set_size =  [1e4,       2e4,        3e4,        4e4,        5e4]

estimate =      [8.33,      23.8,       23.84,      24.2,       24.2]
lower_bound =   [0.0048,    0.00224,    0.00224,    0.03,        0.03]
upper_bound =   [50.44,     115.6,      115.6,      115.6,       115.6]

actual =        [7.79,      14.4,       20.1,       27.48,      35.4]

sb.lineplot(np.array(set_size) * 300, estimate, label='estimate')
# sb.lineplot(network_size, lower_bound, label='lower bound')
# sb.lineplot(network_size, upper_bound, label='upper bound')

sb.lineplot(np.array(set_size) * 300, actual, label='actual', linewidth=3.)

plt.title('Estimation of and actual learning time NN\n(300 features, layers: (8, 100, 100, 100, 100, 1))')
plt.ylabel('Training time [seconds]')
plt.xlabel('Set size [samples x features]')
plt.legend()
# plt.show()
plt.savefig('plots/{!s}.pdf'.format(round(time.time())))