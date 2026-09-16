# Bayesian Modelling for Trading Strategy Selection

A Bayesian statistics project for identifying the most effective trading strategy for an individual stock using probabilistic model comparison and posterior updating.

## Project idea

Trading strategies often provide conflicting signals, and it is difficult for traders to know which one performs better for a given stock. This project addresses that issue by using Bayesian statistics to compare multiple strategy candidates in a systematic and data-driven way.

Instead of relying only on intuition or trial-and-error experience, the project evaluates strategy performance using historical stock data and posterior probabilities. The focus is on identifying the strategy that is most likely to generate profitable market moves for the selected stock.

## Dataset

The project uses daily closing prices of BHARTIARTL (Airtel) over a long historical period, covering approximately 2500 trading days.

The data is sourced from Yahoo Finance and processed in:

- `Stocks data/fetch_data.py`
- `Stocks data/Compute_indicators.py`

The resulting datasets are saved in:

- `Stocks data/Airtel_stocks.csv`
- `Stocks data/Airtel_indicator_values.csv`

## Trading strategies considered

The project compares two trading strategies:

1. 14-RSI + 5-RSI-EMA strategy
2. 9-EMA + 21-EMA crossover strategy

These indicators are computed from the Airtel closing-price series and then used to define daily trading signals.

## Bayesian approach

The project applies a Bayesian framework to estimate the probability of success for each strategy.

- Priors are initialized using Beta distributions.
- Each strategy is evaluated against profitable trading opportunities.
- Posterior means are updated over time as more evidence is observed.
- The strategy with stronger posterior support is considered more effective.

This is implemented in two stages:

### 1. Simplified Bayesian model

The simplified model treats each strategy as having a success probability, updates it across batches of trades, and compares posterior means over time.

Files:

- `Simplified Model/simple_trading_model.py`
- `Simplified Model/simple_testing.py`
- `Simplified Model/results.csv`

### 2. Complex Bayesian model

The more advanced model evaluates strategy behavior over multiple intervals and incorporates richer posterior comparisons and performance tracking.

Files:

- `Complex Model/complex_trading_model.py`
- `Complex Model/Bayesian_testing.py`
- `Complex Model/PPD.py`
- `Complex Model/results_with_weighted_beta.csv`

## Repository structure

- `Complex Model/` — advanced model, testing, and posterior analysis
- `Simplified Model/` — simpler Bayesian comparison pipeline
- `Stocks data/` — stock data download and indicator calculation scripts
- `Plots/` — project plots, notes, and supporting visual assets
- `README.ipynb` — notebook-based project summary
- `README.html` — exported notebook HTML

## Key objective

The goal of the project is to implement a principled Bayesian method for strategy selection so that a trader can choose the more reliable strategy based on evidence, rather than only on intuition or subjective judgment.

## Project summary

This work demonstrates how Bayesian statistics can be used to evaluate and compare trading strategies for a single stock by combining data-driven signal generation with posterior updating. The resulting analysis identifies which strategy is more likely to produce profitable outcomes over time.

## Authors

- Vinay Kar

## Usage

You can explore the scripts and generated outputs in the model folders and the stock-data folder. For a quick project overview, start from the notebook/export files:

- `README.ipynb`
- `README.html`
