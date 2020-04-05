import matplotlib.pyplot as plt
import time
import seaborn as sb

network_size = [1e4, 2e4, 3e4, 4e4, 5e4]

estimate = [11.29, 8.2, 8.2, 12.7, 12.7]
lower_bound = [0.17, 0.1, 0.1, 0.1, 0.1]
upper_bound = [54, 28.5, 28.5, 28.5, 28.5]
actual = [7.28, 14.13, 20.1, 26.15, 33.4]

sb.lineplot(network_size, estimate, label='estimate')
# sb.lineplot(network_size, lower_bound, label='lower bound')
# sb.lineplot(network_size, upper_bound, label='upper bound')

sb.lineplot(network_size, actual, label='actual', linewidth=3.)

plt.title('Estimation of and actual learning time NN\n(8 features, layers: (8, 100, 100, 100, 100, 1))')
plt.legend()
# plt.show()
plt.savefig('plots/{!s}.pdf'.format(round(time.time())))