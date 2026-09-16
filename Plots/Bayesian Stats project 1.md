# Section 3: Multiple Strategy Modelling

Now we will compare the number of profitable trades using multiple strategies. To accomplish this, we use a generalized Beta-Binomial model called Dirichlet-Multinomial model when there are multiple parameters involved.
## Dirichlet-Multinomial Model

The **Dirichlet–Multinomial distribution** is a _compound probability distribution_ that arises when the parameters of a **Multinomial distribution** are themselves random variables drawn from a **Dirichlet distribution**. It is commonly used to model **categorical count data with overdispersion**, where the observed variance exceeds that predicted by a standard multinomial model.
### Definition

Consider a multinomial distribution with parameters $n$ (number of trials) and $\mathbf{p} = (p_1, p_2, \ldots, p_K)$, where $\sum_{i=1}^K p_i = 1$ and $p_i \ge 0$.  
If the vector of probabilities $\mathbf{p}$ is itself drawn from a Dirichlet distribution with concentration parameters $\boldsymbol{\alpha} = (\alpha_1, \alpha_2, \ldots, \alpha_K)$, i.e.
$$p \sim \text{Dirichlet}(\boldsymbol{\alpha})$$
then the resulting marginal distribution over counts $\mathbf{x} = (x_1, x_2, \ldots, x_K)$, where $\sum_{i=1}^K x_i = n$, follows a **Dirichlet–Multinomial distribution**:
$$\mathbf{x} \sim \text{Dirichlet–Multinomial}(n, \boldsymbol{\alpha})$$
### Prior

Assume the category probabilities $\mathbf{p} = (p_1, p_2, \ldots, p_K)$
follow a Dirichlet prior: $$\mathbf{p} \sim \text{Dirichlet}(\boldsymbol{\alpha})$$with parameters $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$.
### Likelihood

Given $\mathbf{p}$, the observed counts $$\mathbf{x} = (x_1, x_2, \ldots, x_K), \quad \sum_{i=1}^K x_i = n$$are drawn from a Multinomial distribution: $$\mathbf{x} \mid \mathbf{p} \sim \text{Multinomial}(n, \mathbf{p})$$The likelihood is: $$P(\mathbf{x} \mid \mathbf{p}) = \frac{n!}{\prod_{i=1}^K x_i!} \prod_{i=1}^K p_i^{x_i}$$
### Posterior

Using Bayes’ rule, the posterior over $\mathbf{p}$ is also Dirichlet: $$\mathbf{p} \mid \mathbf{x} \sim \text{Dirichlet}(\boldsymbol{\alpha} + \mathbf{x}),$$i.e., $$p_i \mid \mathbf{x} \sim \text{Dirichlet}(\alpha_i + x_i) \quad \text{for each } i.$$

## Computing Indicator values

We will compare the following strategies 
- SMA_Crossover
- Momentum_Positive
- RSI_Oversold_Rebound
- RSI_Overbought_Reversal
- Price_Above_EMA9
by modelling them as Dirichlet-Multinomial. But before applying these strategies, they require some indicator values in order to apply them. 

```Python
# ============================================================
# Airtel Stock Data - Indicator Computation Script
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import os

# ------------------------------------------------------------
# Download Airtel stock data
# ------------------------------------------------------------
print("Downloading Airtel stock data...")
data = yf.download('BHARTIARTL.NS', period='2500d', interval='1d')

# --- Flatten MultiIndex if present ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

# --- Handle possible column naming differences ---
close_col = [c for c in data.columns if 'Close' in c][0]

data = data.reset_index()[['Date', close_col]].rename(columns={close_col: 'Close'})
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(inplace=True)

# ------------------------------------------------------------
# Compute Required Indicators
# ------------------------------------------------------------

## --- EMA (for Price above EMA9 strategy) ---
data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()

## --- SMA crossover indicators ---
data['SMA20'] = data['Close'].rolling(20).mean()
data['SMA50'] = data['Close'].rolling(50).mean()

## --- RSI (for Oversold Rebound & Overbought Reversal) ---
delta = data['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14, min_periods=14).mean()
avg_loss = pd.Series(loss).rolling(14, min_periods=14).mean()
rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
data['RSI'] = 100 - (100 / (1 + rs))
data['RSI_MA'] = data['RSI'].rolling(5, min_periods=5).mean()

## --- Momentum (for Momentum Positive strategy) ---
data['Momentum'] = data['Close'] - data['Close'].shift(3)

## --- Next-day Close (for future trade labeling if needed) ---
data['Close_next'] = data['Close'].shift(-1)

# ------------------------------------------------------------
# Keep Only Required Columns
# ------------------------------------------------------------
cols_to_keep = [
    'Date',
    'Close',
    'EMA9',
    'SMA20',
    'SMA50',
    'RSI',
    'RSI_MA',
    'Momentum',
    'Close_next'
]

data = data[cols_to_keep].dropna().reset_index(drop=True)

# ------------------------------------------------------------
# Save the final filtered indicator file
# ------------------------------------------------------------
output_file = 'Airtel_selected_strategies.csv'
data.to_csv(output_file, index=False)

print(f"Saved {output_file} with only the selected strategy indicators.")
```
### Data Acquisition and Preparation

The code begins by downloading **daily historical stock data** for _Bharti Airtel_ using the `yfinance` library:
```Python
data = yf.download('BHARTIARTL.NS', period='2500d', interval='1d')
```
- The data covers the past **~2500 trading days** (roughly 10 years).
- Only the **Date** and **Close price** columns are retained for analysis.
- Any missing or non-numeric data is removed to ensure accuracy.
### Computation of Technical Indicators

Several important **technical indicators** are computed to capture different aspects of stock price behavior:
#### (a) Exponential Moving Average (EMA9)

The **9-day Exponential Moving Average** smoothens out short-term price fluctuations and highlights the underlying trend. It is calculated using: $$\text{EMA}_t = \alpha \cdot \text{Close}_t + (1 - \alpha) \cdot \text{EMA}_{t-1}, \quad \text{where } \alpha = \frac{2}{9 + 1}$$
- Used in **“Price above EMA9”** strategy to detect short-term bullish or bearish trends.
- A price above EMA9 often signals upward momentum.
#### (b) Simple Moving Averages (SMA20 and SMA50)

Two **Simple Moving Averages** are computed:
- **SMA20:** 20-day average — short-term trend
- **SMA50:** 50-day average — medium-term trend
They are defined as:
$$\text{SMA}_t(k) = \frac{1}{k} \sum_{i=0}^{k-1} \text{Close}_{t-i}$$
- The **SMA crossover** (e.g., SMA20 > SMA50) is often used to detect **trend reversals** or **trend confirmations**.
#### (c) Relative Strength Index (RSI)

The **Relative Strength Index (RSI)** is a momentum oscillator that measures the magnitude of recent price changes to evaluate **overbought** or **oversold** conditions.  
It is computed over a 14-day window as: $$\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}}$$
- **High RSI (>70):** indicates overbought conditions (possible price reversal downwards).
- **Low RSI (<30):** indicates oversold conditions (possible price rebound).
- A **5-day moving average of RSI (RSI_MA)** is also calculated to smooth RSI values.
#### (d) Momentum

Momentum measures the rate of price change over a short period (3 days here): $$\text{Momentum}_t = \text{Close}_t - \text{Close}_{t-3}$$
- A **positive momentum** implies upward movement and buying pressure.
- A **negative momentum** indicates downward movement or selling pressure.
- Useful in **momentum trading strategies**.
#### (e) Next-Day Close

A shifted column is created: $$\text{Close}_{t+1} = \text{Close shifted by one day}.$$This allows for **future return labeling**, useful in supervised learning or strategy back-testing.
## Multiple Strategy Evaluation

Now once we have the technical indicators, we now move on to evaluating the following strategies:  

| *Strategy*                  | *Key Indicators* | *Market Logic*                                      |
| --------------------------- | ---------------- | --------------------------------------------------- |
| **SMA_Crossover**           | SMA20, SMA50     | Detects trend changes via moving average crossovers |
| **Momentum_Positive**       | Momentum         | Trades continuation of short-term price strength    |
| **RSI_Oversold_Rebound**    | RSI, RSI_MA      | Buys when price is oversold and starts rebounding   |
| **RSI_Overbought_Reversal** | RSI, RSI_MA      | Sells when price is overbought and starts reversing |
| **Price_Above_EMA9**        | EMA9             | Follows short-term trend — buy when price > EMA9    |

For each strategy, a **profit indicator** is computed as: $$\text{profit} = \begin{cases} 1, & \text{if signal = 1 and Close}_{t+1} > \text{Close}_t \\ 0, & \text{otherwise} \end{cases}$$This creates binary outcomes (`1 = profitable`, `0 = not profitable`), forming the **observed data** for Bayesian updating.

The data is divided into:
- **Training Set (80%)** - used for Bayesian learning (posterior updates)
- **Testing Set (20%)** - used for evaluating strategy generalization and performance stability.
This ensures that the posterior estimates are learned on historical data and validated on unseen data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================================================
# Load precomputed indicator data
# ====================================================
file_path = 'Airtel_selected_strategies.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError("Run DM_five_indicators.py first to create Airtel_selected_strategies.csv")

data = pd.read_csv(file_path)
print(f"Loaded {len(data)} rows from {file_path}")

# ====================================================
# Verify necessary columns exist
# ====================================================
required_cols = ['Close', 'Close_next', 'SMA20', 'SMA50', 'Momentum', 'RSI', 'EMA9']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# ====================================================
# Define 5 trading strategies
# ====================================================
strategies = {
    "SMA_Crossover": (data['SMA20'] > data['SMA50']).astype(int),
    "Momentum_Positive": (data['Momentum'] > 0).astype(int),
    "RSI_Oversold_Rebound": ((data['RSI'] < 30) & (data['Momentum'] > 0)).astype(int),
    "RSI_Overbought_Reversal": ((data['RSI'] > 70) & (data['Momentum'] < 0)).astype(int),
    "Price_Above_EMA9": (data['Close'] > data['EMA9']).astype(int)
}

# ====================================================
# Compute profitable trades (1 = profitable, 0 = not)
# ====================================================
for strat, signal in strategies.items():
    data[f'{strat}_profit'] = ((signal == 1) & (data['Close_next'] > data['Close'])).astype(int)

data = data.dropna().reset_index(drop=True)

# ====================================================
# Split data into Training (80%) and Testing (20%)
# ====================================================
split_index = int(0.8 * len(data))
train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()
print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

# ====================================================
# Dirichlet–Multinomial Model (Training Phase)
# ====================================================
def run_dirichlet_multinomial(data, strategy_names, interval_size=50, trades=20):
    K = len(strategy_names)
    alpha = np.ones(K)  # prior (uniform)
    num_intervals = len(data) // interval_size
    if num_intervals == 0:
        raise ValueError("Not enough data for one interval. Try smaller interval_size.")

    results = []
    for idx in range(num_intervals):
        start, end = idx * interval_size, (idx + 1) * interval_size
        interval = data.iloc[start:end]
        step = max(1, len(interval) // trades)
        trade_idx = np.arange(0, len(interval), step)[:trades]
        interval = interval.iloc[trade_idx]

        successes = np.array([interval[f'{name}_profit'].sum() for name in strategy_names])
        alpha += successes
        posterior_means = alpha / np.sum(alpha)

        record = {'Interval': idx}
        for i, name in enumerate(strategy_names):
            record[f'{name}_posterior'] = posterior_means[i]
        results.append(record)

    results_df = pd.DataFrame(results)
    final_posterior = alpha / np.sum(alpha)
    return results_df, final_posterior, alpha

# ====================================================
# Run Dirichlet–Multinomial on TRAINING DATA
# ====================================================
strategy_names = list(strategies.keys())
train_results, train_final_post, posterior_alpha = run_dirichlet_multinomial(train_data, strategy_names)

# ====================================================
# Compute TESTING averages for each strategy
# ====================================================
test_avg_profit = [test_data[f'{name}_profit'].mean() for name in strategy_names]
train_avg_posterior = train_final_post  # posterior mean from training

# ====================================================
# Create output folder
# ====================================================
os.makedirs('Dirichlet_Model', exist_ok=True)

# ====================================================
# Plot 1 — Training Posterior Means (Dirichlet Multinomial)
# ====================================================
plt.figure(figsize=(10, 6))
for name in strategy_names:
    plt.plot(train_results['Interval'], train_results[f'{name}_posterior'], label=name)

plt.xlabel('Interval')
plt.ylabel('Posterior Mean Probability')
plt.title('Training Posterior Means — Dirichlet Multinomial Model')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("Dirichlet_Model/Training_Posterior_Means.png")
plt.show()

# ====================================================
# Plot 2 — Testing Data Means (Average Profitability)
# ====================================================
plt.figure(figsize=(8, 5))
plt.bar(strategy_names, test_avg_profit, color='lightcoral')
plt.xticks(rotation=30)
plt.ylabel('Average Profitability')
plt.title('Mean of Profitable Trades — Testing Data')
plt.tight_layout()
plt.savefig("Dirichlet_Model/Testing_Average_Profits.png")
plt.show()

# ====================================================
# Plot 3 — Compare Training Posterior vs Testing Means
# ====================================================
x = np.arange(len(strategy_names))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, train_avg_posterior, width, label='Training Posterior Means', color='skyblue')
plt.bar(x + width/2, test_avg_profit, width, label='Testing Average Profits', color='lightcoral')
plt.xticks(x, strategy_names, rotation=30)
plt.ylabel('Probability / Average Profitability')
plt.title('Comparison — Training Posterior vs Testing Mean Profits')
plt.legend()
plt.tight_layout()
plt.savefig("Dirichlet_Model/Train_vs_Test_Comparison.png")
plt.show()

# ====================================================
# Print Summary and Comparison
# ====================================================
summary_df = pd.DataFrame({
    'Training_Posterior_Mean': train_avg_posterior,
    'Testing_Average_Profit': test_avg_profit
}, index=strategy_names)
summary_df['Difference'] = summary_df['Testing_Average_Profit'] - summary_df['Training_Posterior_Mean']

print("All plots saved in Dirichlet_Model/\n")
print("Average Comparison of Profitable Trades:")
print(summary_df.round(4))

# ====================================================
# 🧮 Monte Carlo Posterior Predictive Simulation (SMA_Crossover)
# ====================================================
print("\nRunning Monte Carlo Posterior Predictive Simulation for SMA_Crossover...\n")

N_SAMPLES = 10000        # Number of posterior draws
N_FUTURE_TRADES = 100    # Simulate 100 future trades
STRATEGY = "SMA_Crossover"

# Draw from Dirichlet posterior
dirichlet_samples = np.random.dirichlet(posterior_alpha, size=N_SAMPLES)

# Extract SMA_Crossover probabilities
s_idx = strategy_names.index(STRATEGY)
p_samples = dirichlet_samples[:, s_idx]

# Simulate future trades for each sampled p
success_count = 0
for p in p_samples:
    future_trades = np.random.binomial(n=N_FUTURE_TRADES, p=p)
    if future_trades / N_FUTURE_TRADES > 0.5:
        success_count += 1

# Compute estimated probability
prob_success_over_50 = success_count / N_SAMPLES

# Display results
print(f"Estimated P({STRATEGY} profitable > 50%) ≈ {prob_success_over_50:.4f}")

# --- Plot histogram of posterior samples (p_SMA) ---
plt.figure(figsize=(8,5))
plt.hist(p_samples, bins=40, color='skyblue', edgecolor='black', density=True)
plt.axvline(0.5, color='red', linestyle='--', label='50% threshold')
plt.title(f"Posterior Samples of {STRATEGY} Profitability (Monte Carlo)")
plt.xlabel("Profitability Probability (p_SMA)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f"Dirichlet_Model/{STRATEGY}_Profitability_Distribution.png")
plt.show()
```
### Training Data

During training, the data is processed in **intervals** (batches) of trades.  
For each interval of $n$ trades (in our case, $n = 20$), we observe how many trades were profitable for each of the $K = 5$ strategies.
#### Prior

During the training phase, we initially start with a non-informative Dirichlet prior: $$\mathbf{p} \sim \text{Dirichlet}(\boldsymbol{\alpha}),\qquad \boldsymbol{\alpha}=(\alpha_1,\dots,\alpha_5)$$ with $\alpha_i = 1$ for $1 \leq i \leq 5$. This represents a **uniform prior belief**, meaning all strategies are initially assumed equally likely to be profitable.
#### Likelihood

For a batch of $n$ trials (trades where one strategy could be counted per trial), the vector of observed successes $\mathbf{x}=(x_1,\dots,x_5)$ is modeled as $$\mathbf{x}\mid \mathbf{p}\ \sim\ \text{Multinomial}\big(n,\mathbf{p}\big)$$
where $x_i$ denotes the number of profitable trades for strategy $i$ within that batch. This corresponds to computing the **sum of profitable trades** per strategy within each 20-trade interval.
#### Posterior 

Because the Dirichlet is conjugate to the Multinomial, the posterior is analytically simple:
$$\mathbf{p}\mid \mathbf{x}\ \sim\ \text{Dirichlet}(\boldsymbol{\alpha}+\mathbf{x})$$
Hence the _posterior mean_ (a point summary often used) is $$\mathbb{E}[p_i\mid\mathbf{x}] = \frac {\alpha_i + x_i}{\sum_{j=1}^5 (\alpha_j + x_j)}.$$
#### Sequential Updating Across Time

This updating process is repeated sequentially across multiple time intervals (each representing a block of 20 trades).  
After each interval:
1. The **posterior** from the previous batch becomes the **new prior**.
2. The model incorporates the next batch of data to update $\boldsymbol{\alpha}$.
This allows the Bayesian model to **adapt to changing market conditions** and track the evolving profitability of each trading strategy.
At each step, the **posterior mean** values: $$E[p_i \mid \text{data}] = \frac{\alpha_i}{\sum_j \alpha_j}$$represent the **probability that a strategy will be profitable**, given all observations up to that point.

![jpg](README_files/Training_Posterior_Means.jpg)
This line plot shows how the **posterior mean probabilities** of profitability for each strategy evolve over successive training intervals (each interval representing 20 trades per batch).
### Testing Data 

On the testing dataset:
- The **average profit rate** for each strategy is computed as the mean of its profit indicators.
- These testing averages are compared against the **posterior means** obtained from training.
This step checks whether the Bayesian model’s learned beliefs about each strategy’s success probability align with their actual performance on unseen data.

![png](README_files/Plot_3.jpg)

This bar chart displays the **mean profitability** of each strategy on the **testing dataset**. Each bar represents one strategy’s **average profit rate** on test data, computed as the fraction of profitable trades in the testing period.

![png](README_files/plot_2.jpg)This figure shows a side-by-side comparison between the **posterior mean probabilities** obtained during the training phase and the **average profitability** observed in the testing phase for each trading strategy.
For every strategy $i$:
- The **blue bar** represents the model’s estimated profitability, $\mathbb{E}[p_i \mid \text{training data}]$, derived from the Dirichlet–Multinomial posterior.
- The **red bar** represents the **empirical mean profit**, $\bar{p}_i^{\text{test}}$​, computed from unseen (testing) data.
Here, $\bar{p}_i^{\text{test}}$ denotes the **average observed profitability** of strategy $i$ across all test trades: $$\bar{p}_i^{\text{test}} = \frac{1}{n_{\text{test}}} \sum_{t=1}^{n_{\text{test}}} p_{i,t}, \quad p_{i,t} = \begin{cases} 1, & \text{if trade } t \text{ was profitable} \\ 0, & \text{otherwise.} \end{cases}$$Thus, $\bar{p}_i^{\text{test}}$​ represents how frequently a strategy yielded profits on unseen data.

Ideally, the two bars for each strategy should be close in height, indicating that the **posterior estimates** generalize well to new data. Large discrepancies suggest potential **model bias** or **instability**, such as when a strategy performs strongly during training but weakly during testing. 
In our case, the posterior estimates align closely with the observed testing profits for most strategies particularly **SMA_Crossover**, **Momentum_Positive**, and **Price_Above_EMA9** indicating that the Bayesian model provides **reasonably accurate and stable predictions** of strategy profitability on unseen data.

---
## Monte Carlo Posterior-Predictive Analysis

The Dirichlet–Multinomial model gives an analytically convenient posterior for the vector of strategy-profit probabilities $\mathbf{p}=(p_1,\dots,p_K)$. The posterior mean is useful, but it hides uncertainty and does not directly answer predictive questions such as:

> **What is the probability that strategy `SMA_Crossover` will be profitable in more than 50% of future trades?**

Computing that probability analytically from the posterior predictive distribution (PPD) is possible in simple conjugate models but becomes cumbersome for vector-valued Dirichlet–Multinomial models, so we use **Monte Carlo simulation** to approximate it.
### Algorithm
We wish to estimate the probability that the **SMA Crossover** strategy will be profitable in more than $50$% of future trades.
- **Posterior sampling:**  
    Draw samples of the profitability vector $\mathbf{p}$ repeatedly from the Dirichlet posterior $p \sim \text{Dirichlet}(\boldsymbol{\alpha}_{\text{posterior}})$
- **Predictive simulation:**  
    For each draw, extract $p_{\text{SMA}}$ and simulate the number of profitable trades in $N$ future trials:  $$X_{\text{SMA}} \sim \text{Binomial}(N, p_{\text{SMA}})$$
- **Estimate target probability:**  
    Count how many simulated futures satisfy $X_{\text{SMA}}/N > 0.5$. The Monte Carlo estimate $$\widehat{P} = \frac{1}{M}\sum_{m=1}^{M}\mathbf 1\!\left\{X_{\text{SMA}}^{(m)}/N>0.5\right\}$$approximates $\Pr(\text{SMA Crossover profitable} > 50\%)$

Running the above algorithm gives us the following output which is also evident from the graph below
```python
🔹 Estimated P(SMA_Crossover profitable > 50%) ≈ 0.0008
```
#### Why this works

- Sampling $p_{\text{SMA}}$ from the Dirichlet posterior represents uncertainty in the true profitability.
- Simulating Binomial outcomes for each $p_{\text{SMA}}$​ adds uncertainty in **future market realizations.
- Repeating the process many times approximates the full posterior-predictive distribution; the proportion of simulations above $50$ % gives the desired probability.


Graph representing the posterior distribution of the probability that an SMA (Simple Moving Average) crossover trading strategy is profitable, based on Monte Carlo simulations.

![jpg](Monte_Carlo.jpg)