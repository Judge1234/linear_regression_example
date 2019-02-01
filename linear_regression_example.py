import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


SAMPLES             = 35
SLOPE               = 2.75
Y_INT               = 3.5                                                                                                                                                                                                                                                             
NOISE_LOW_BOUND     = -7
NOISE_HIGH_BOUND    = 5
XS_LOW_BOUND        = -10
XS_HIGH_BOUND       = 7
COEF_PREDICT        = []
RSQUARED            = []
SS                  = []


def linfunc(m, x, b):
    noise = np.random.uniform(NOISE_LOW_BOUND, NOISE_HIGH_BOUND)
    y = (m * x) + b + noise
    return y


xs = np.linspace(XS_LOW_BOUND, XS_HIGH_BOUND, SAMPLES)
ys = np.array([linfunc(SLOPE, x, Y_INT) for x in xs])
sample_data = np.array([i for i in zip(xs, ys)])
centroid = round(np.average(xs), 2), round(np.average(ys), 2)


def linreg(data):
    xs = data[:, 0]
    ys = data[:, 1]
    xbar = np.average(xs)
    ybar = np.average(ys)
    centroid = plt.scatter(xbar, ybar, s=250, marker='+')
    
    beta1_numerator = sum([(xs[i] - xbar) * (ys[i] - ybar) for i in range(len(xs))])
    beta1_denominator = sum((xs[i] - xbar) ** 2 for i in range(len(xs)))
    beta1 = beta1_numerator / beta1_denominator
    beta0 = ybar - (beta1 * xbar)
    COEF_PREDICT.extend([beta0, beta1])
    
    yhats = [beta0 + (beta1 * xs[i]) for i in range(len(xs))]
    
    SST = sum((ys - ybar) ** 2)
    SSE = sum((ys - yhats) ** 2)
    SSR = SST - SSE
    SS.extend([round(SST, 4), round(SSE, 4), round(SSR, 4)])
    RSQUARED.append(SSR / SST)
    
    return yhats


sns.set()
fig, ax = plt.subplots(figsize=(14, 8))
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
fontsize = 14
plt.plot(xs, linreg(sample_data), c='black')
plt.scatter(xs, ys)

display_R2 = f'R-Squared: {round(RSQUARED[0], 4)}\n'
display_PS = f'Predicted slope: {round(COEF_PREDICT[1], 4)}\n'
display_PY = f'Predicted y-int: {round(COEF_PREDICT[0], 4)}'
display_AS = f'Actual slope: {SLOPE}\n'
display_AY = f'Actual y-int: {Y_INT}'
display_CT = f'Centroid: {centroid}\n'
display_SS = [f'SST: {SS[0]}\n', f'SSE: {SS[1]}\n', f'SSR: {SS[2]}']


plt.title('Linear Regression Example')
reg_formula = r'$\^y = \beta_0 + \beta_1x_i$'
beta0_formula = r'$\beta_0 = \bar y - \beta_1\bar x$' + '\n'
beta1_formula = r'$\beta_1 = \frac{\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^n (x_i - \bar x)^2}$'

ax.text(int(np.average(xs) - 4), min(ys), display_R2 + display_PS + display_PY, fontsize=fontsize)
ax.text(int(np.average(xs)), min(ys), display_CT + display_AS + display_AY, fontsize=fontsize)
ax.text(int(np.average(xs) + 4), min(ys), display_SS[0] + display_SS[1] + display_SS[2], fontsize=fontsize)
ax.text(int(np.average(xs) - 6), max(ys) * 0.6, beta0_formula + beta1_formula, fontsize=18)
main_legend = ax.legend([reg_formula, 'Centroid', 'Samples'], loc=2)
ax.add_artist(main_legend)

plt.show()







