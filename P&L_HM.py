from statistics import stdev
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# P&L Calculation for Single Leg
def calculate_leg_pnl(underlying_prices, leg):
    sp = leg["strike_price"]
    ip = leg["initial_price"]
    cs = leg["contract_size"]
    t = leg["option_type"]
    if t == "Call":
        payoff = np.maximum(underlying_prices - sp, 0) - ip
    elif t == "Put":
        payoff = np.maximum(sp - underlying_prices, 0) - ip
    else:
        raise ValueError("Invalid option type: must be 'Call' or 'Put'")
    return cs * payoff * 100  # Adjusted for contract size only

# Total P&L Calculation (All Legs)
def calculate_total_pnl(underlying_prices, legs):
    total_pnl = np.zeros_like(underlying_prices, dtype=float)
    leg_pnls = []
    for leg in legs:
        leg_pnl = calculate_leg_pnl(underlying_prices, leg)
        total_pnl += leg_pnl
        leg_pnls.append(leg_pnl)
    return total_pnl, leg_pnls

# Break-even Points Calculation
def find_break_even_points(underlying_prices, total_pnl):
    indices = np.where(np.diff(np.sign(total_pnl)))[0]
    break_even = []
    for idx in indices:
        x1, x2 = underlying_prices[idx], underlying_prices[idx + 1]
        y1, y2 = total_pnl[idx], total_pnl[idx + 1]
        f = interp1d([y1, y2], [x1, x2])
        break_even.append(f(0))
    return np.array(break_even)

# Heston Model Simulation
def heston_mcs(S0, v0, mu, kappa, theta, sigma_v, rho, T, steps, simulations):
    dt = T / steps
    S = np.zeros((simulations, steps + 1))
    v = np.zeros((simulations, steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, steps + 1):
        Z1 = np.random.normal(0, 1, simulations)
        Z2 = np.random.normal(0, 1, simulations)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v[:, t] = v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma_v * np.sqrt(np.maximum(v[:, t - 1], 0)) * np.sqrt(dt) * Z2
        v[:, t] = np.maximum(v[:, t], 0)

        S[:, t] = S[:, t - 1] * np.exp((mu - 0.5 * np.maximum(v[:, t], 0)) * dt + np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z1)

    return S, v

# Parameters for Heston Model
v0 = 0.2**2
kappa = 2.0
theta = v0
sigma_v = 0.3
rho = -0.5
steps = 252  # 1 year of daily steps

# Monte Carlo Parameters
num_simulations = 10000
drift = 0.05
days_to_expiry = 3
T = days_to_expiry / 252
underlying_price = 589.49
confidence_level = 0.95

# Option Legs
legs = [
    {"strike_price": 570, "initial_price": 3.2, "contract_size": -1, "option_type": "Put"},
    {"strike_price": 580, "initial_price": 6.5, "contract_size": 1, "option_type": "Put"},
    {"strike_price": 590, "initial_price": 5.5, "contract_size": 1, "option_type": "Call"},
    {"strike_price": 600, "initial_price": 2.7, "contract_size": -1, "option_type": "Call"},
]

# Simulate Prices Using Heston Model
simulated_prices, simulated_variances = heston_mcs(
    S0=underlying_price, v0=v0, mu=drift, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho,
    T=T, steps=steps, simulations=num_simulations
)

# Final Prices and P&L Calculations
final_prices = simulated_prices[:, -1]
simulated_pnls = calculate_total_pnl(final_prices, legs)[0]
downside_dev = np.std(simulated_pnls[simulated_pnls < 0])

# Metrics Calculations
max_profit = np.max(simulated_pnls)
max_loss = np.min(simulated_pnls)
reward_risk_ratio = abs(max_profit / max_loss) if max_loss != 0 else None
expected_pnl = np.mean(simulated_pnls)
prob_of_profit = (np.sum(simulated_pnls > 0) / num_simulations) * 100
std_dev_pnls = np.std(simulated_pnls)
sharpe_ratio = (expected_pnl - drift) / std_dev_pnls
sortino_ratio = (expected_pnl - drift) / downside_dev

# VAR Calculation
def calculate_var(pnl_simulations, confidence_level):
    alpha = 1 - confidence_level
    return np.percentile(pnl_simulations, alpha * 100)

# CVAR Calculation
def calculate_cvar(pnl_simulations, confidence_level):
    var = calculate_var(pnl_simulations, confidence_level)
    extreme_losses = pnl_simulations[pnl_simulations <= var]
    return extreme_losses.mean()

var = calculate_var(simulated_pnls, confidence_level)
cvar = calculate_cvar(simulated_pnls, confidence_level)

# Display Metrics
print("Max Profit:", round(max_profit, 2))
print("Max Loss:", round(max_loss, 2))
print("Reward to Risk Ratio:", round(reward_risk_ratio, 2))
print("Expected P&L:", round(expected_pnl, 2))
print(f"Probability of Profit: {prob_of_profit}%")
print("Standard Deviation (Volatility):", round(std_dev_pnls, 2))
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Sortino Ratio:", round(sortino_ratio, 2))
print("VaR:", round(var, 2))
print("CVaR:", round(cvar, 2))
