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
        # Linear interpolation
        f = interp1d([y1, y2], [x1, x2])
        break_even.append(f(0))  # P&L = 0 is the break-even point
    return np.array(break_even)

# Monte Carlo Simulation
def mcs(legs, underlying_price, volatility, drift, days_to_expiry, num_simulations):
    dt = 1/252
    num_steps = days_to_expiry
    simulated_prices = np.zeros((num_simulations, num_steps))
    simulated_prices[:, 0] = underlying_price
    for step in range(1, num_steps):
        random = np.random.normal(0, 1, num_simulations)
        simulated_prices[:, step] = simulated_prices[:, step - 1] * np.exp((drift - 0.5 * volatility ** 2) *
                                                                           dt + volatility * np.sqrt(dt) * random
                                                                          )
    final_prices = simulated_prices[:, -1]
    total_pnls = calculate_total_pnl(final_prices, legs)[0]
    return total_pnls

# Parameters for Monte Carlo
num_simulations = 10000
volatility = 0.20
drift = 0.05
days_to_expiry = 3
underlying_price = 589.49
confidence_level = 0.95

# Parameters for Option Legs
legs = [{"strike_price": 570, "initial_price": 3.2, "contract_size": -1, "option_type": "Put"},
        {"strike_price": 580, "initial_price": 6.5, "contract_size": 1, "option_type": "Put"},
        {"strike_price": 590, "initial_price": 5.5, "contract_size": 1, "option_type": "Call"},
        {"strike_price": 600, "initial_price": 2.7, "contract_size": -1, "option_type": "Call"},
       ]

# Current Total P&L Calculation
def current_pnl(leg):
    underlying_price = 589.49
    option_type = leg["option_type"]
    contracts = leg["contract_size"]
    strike_price = leg["strike_price"]
    initial_price = leg["initial_price"]
    if option_type == "Call":
        intrinsic_value = max(underlying_price - strike_price, 0)
    elif option_type == "Put":
        intrinsic_value = max(strike_price - underlying_price, 0)
    else:
        raise ValueError("Invalid option type")
    pnl_per_share = (intrinsic_value - initial_price) * np.sign(contracts)
    current_pnl = pnl_per_share * abs(contracts) * 100
    return current_pnl
pnl_legs = [current_pnl(leg) for leg in legs]
current_pnl = sum(pnl_legs)

# VAR Calculation
def calculate_var(pnl_simulations, confidence_level):
    alpha = 1 - confidence_level
    var1 = np.percentile(pnl_simulations, alpha * 100)
    return var1

# CVAR Calculation
def calculate_cvar(pnl_simulations, confidence_level):
    var2 = calculate_var(pnl_simulations, confidence_level)
    extreme_losses = pnl_simulations[pnl_simulations <= var2]
    cvar1 = extreme_losses.mean()
    return cvar1

# Parameters for X-Axis Range
underlying = 589.49  # Current underlying price
step = 0.25  # Step size
underlying_prices = np.arange(underlying - 100 * step, underlying + 100 * step, step)

# Total/Leg P&L Calculations
total_pnl, leg_pnls = calculate_total_pnl(underlying_prices, legs)

# Run Monte Carlo
simulated_pnls = mcs(legs, underlying_price, volatility, drift, days_to_expiry, num_simulations)
downside_dev = np.std(simulated_pnls[simulated_pnls < 0])

# Metrics Calculations
max_profit = np.max(total_pnl)
max_loss = np.min(total_pnl)
reward_risk_ratio = abs(max_profit / max_loss) if max_loss != 0 else None
break_even = find_break_even_points(underlying_prices, total_pnl)
expected_pnl = np.mean(simulated_pnls)
prob_of_profit = (np.sum(simulated_pnls > 0) / num_simulations) * 100
std_dev_pnls = np.std(simulated_pnls)
sharpe_ratio = (expected_pnl - drift) / std_dev_pnls
sortino_ratio = (expected_pnl - drift) / downside_dev
var = calculate_var(simulated_pnls, confidence_level)
cvar = calculate_cvar(simulated_pnls, confidence_level)

# Creating an Interactive Graph
fig = go.Figure()

# Green Shaded Area (Positive P&L)
for alpha in np.linspace(0.05, 0.3, 5):  # Create 5 layers with increasing opacity
    fig.add_trace(
        go.Scatter(x=underlying_prices,
                   y=np.where(total_pnl >= 0, total_pnl * alpha / 0.3, 0),  # Scale by alpha
                   fill='tozeroy',  # Fill to the x-axis
                   fillcolor=f'rgba(0, 255, 0, {alpha})',  # Green fill with varying opacity
                   line=dict(color='rgba(0, 255, 0, 0)', width=0),  # No visible line
                   showlegend=False  # Hide legend for these layers
                  )
                 )

# Red Shaded Area (Negative P&L)
for alpha in np.linspace(0.05, 0.3, 5):  # Create 5 layers with increasing opacity
    fig.add_trace(
        go.Scatter(x=underlying_prices,
                   y=np.where(total_pnl <= 0, total_pnl * alpha / 0.3, 0),  # Scale by alpha
                   fill='tozeroy',  # Fill to the x-axis
                   fillcolor=f'rgba(255, 0, 0, {alpha})',  # Red fill with varying opacity
                   line=dict(color='rgba(255, 0, 0, 0)', width=0),  # No visible line
                   showlegend=False  # Hide legend for these layers
                  )
                 )

# Green Line/Red Line
positive_pnl = np.where(total_pnl >= 0, total_pnl, np.nan)
negative_pnl = np.where(total_pnl < 0, total_pnl, np.nan)

# Positive P&L Line Colour
fig.add_trace(
    go.Scatter(x=underlying_prices,
               y=positive_pnl,
               mode='lines',
               name='Positive P&L',
               line=dict(color='green', width=3)
              )
             )

# Negative P&L Line Colour
fig.add_trace(
    go.Scatter(x=underlying_prices,
               y=negative_pnl,
               mode='lines',
               name='Negative P&L',
               line=dict(color='red', width=3)
              )
             )

# Highlight Break-even Points
for bep in break_even:
    fig.add_trace(
        go.Scatter(x=[bep],
                   y=[0],
                   mode='markers',
                   name='Break-even',
                   marker=dict(color='black', size=8, symbol='x')
                  )
                 )

# Graph Parameters
fig.update_layout(title="P&L Calculator",
                  xaxis_title="Underlying Price ($)",
                  yaxis_title="P&L ($)",
                  template="plotly_dark",
                  height=600,
                  title_x=0.5,
                  title_y=0.93,
                  title_font=dict(size=18,
                                  color='white',
                                  weight='bold',
                                 )
                 )

# Displaying Graph
fig.show()

# Printing Metrics
print("Max Profit:", round(max_profit, 2))
print("Max Loss:", round(max_loss, 2))
print("Reward to Risk Ratio:", round(reward_risk_ratio, 2))
print("Break-even Points:", break_even)
print("Current P&L:", round(current_pnl, 2))
print("Expected P&L:", round(expected_pnl, 2))
print(f"Probability of Profit: {prob_of_profit}%")
print("Standard Deviation (Volatility):", round(std_dev_pnls, 2))
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Sortino Ratio:", round(sortino_ratio, 2))
print("Var:", round(var, 2))
print("CVar:", round(cvar, 2))
