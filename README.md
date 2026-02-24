# Data Generation using Modelling & Simulation for Machine Learning

> **Assignment**: Generate synthetic data via simulation, then train and compare ML models on it.

---

## Table of Contents
1. [Overview](#overview)
2. [Simulation Tool: SimPy](#simulation-tool-simpy)
3. [M/M/c Queue Model](#mmc-queue-model)
4. [Parameters & Bounds](#parameters--bounds)
5. [Methodology](#methodology)
6. [Results Table](#results-table)
7. [Result Graphs](#result-graphs)
8. [Conclusion](#conclusion)
9. [How to Run](#how-to-run)

---

## Overview

This project demonstrates a complete **Simulation → Data Collection → Machine Learning** pipeline:

1. A real-world **queuing system** (e.g., a bank, hospital, call center) is modelled using SimPy
2. **1,000 unique simulations** are run with randomly sampled parameters
3. **10 ML regression models** are trained to predict average customer wait time
4. Models are compared across RMSE, MAE, R², and Cross-Validation metrics

**ML Task**: Predict `avg_wait_time` (the mean time a customer waits before service begins) given queue configuration parameters.

---

## Simulation Tool: SimPy

| Property | Detail |
|----------|--------|
| **Name** | SimPy |
| **Type** | Discrete-Event Simulation (DES) |
| **Language** | Python |
| **Install** | `pip install simpy` |
| **Documentation** | https://simpy.readthedocs.io/ |
| **Wikipedia** | https://en.wikipedia.org/wiki/SimPy |

SimPy is a process-based discrete-event simulation library. It allows you to model real-world systems where entities (customers, packets, jobs) arrive, wait in queues, consume resources (servers, machines, bandwidth), and depart.

**Why SimPy?**
- Pure Python, easy to install and use
- Widely used in operations research, logistics, healthcare, and network modelling
- Supports complex conditional logic (priority queues, reneging, preemption)
- Validated against analytical queueing theory results

---

## M/M/c Queue Model

The **M/M/c queue** (also called Erlang-C model) is one of the fundamental models in queueing theory:

```
Customers → [Arrival Process] → [Queue] → [c Servers] → Departure
                Poisson              FIFO    Exponential
                                            Service Times
```

- **First M**: Markovian (Poisson) arrivals — exponential inter-arrival times
- **Second M**: Markovian service — exponential service durations
- **c**: Number of parallel servers

**System stability condition**: `ρ = λ/(c·μ) < 1`  
(arrival rate must be less than total service capacity)

### Theoretical Predictions
- As traffic intensity ρ → 1, wait times grow **exponentially** (non-linear)
- Adding more servers dramatically reduces wait times near saturation
- This non-linearity makes it a great test case for ML — linear models will struggle

---

## Parameters & Bounds

### Input Parameters (Features)

| Parameter | Symbol | Lower Bound | Upper Bound | Type | Description |
|-----------|--------|-------------|-------------|------|-------------|
| `arrival_rate` | λ | **0.5** | **4.0** | Float | Customers arriving per time unit |
| `service_rate` | μ | **1.0** | **5.0** | Float | Services completed per server per time unit |
| `num_servers` | c | **1** | **5** | Integer | Number of parallel servers |
| `traffic_intensity` | ρ | derived | derived | Float | λ / (c·μ) — system load |

**Rationale for bounds:**
- `arrival_rate` ≥ 0.5: Need enough arrivals to gather reliable statistics in 500 time units
- `service_rate` ≥ 1.0: Prevents servers from being infinitely slow
- `num_servers` 1–5: Covers single-counter to small call-center scale
- Upper bounds set to allow a diverse mix of stable (ρ < 1) and overloaded (ρ ≥ 1) systems

### Output Variables (Simulation Records)

| Variable | Description | Used As |
|----------|-------------|---------|
| `avg_wait_time` | Mean time in queue before service | **ML Target** |
| `avg_queue_length` | Mean number waiting in queue | Feature / Analysis |
| `throughput` | Customers served per time unit | Analysis |
| `utilization` | Server capacity utilization fraction | Analysis |

---

## Methodology

### Step 1–2: Install & Explore SimPy
```python
!pip install simpy
import simpy
env = simpy.Environment()
server = simpy.Resource(env, capacity=2)
```

### Step 3: Parameter Sampling
Each simulation draws parameters uniformly at random:
```python
arrival_rate = np.random.uniform(0.5, 4.0)   # λ
service_rate = np.random.uniform(1.0, 5.0)   # μ
num_servers  = np.random.randint(1, 6)        # c ∈ {1,2,3,4,5}
```

### Step 4: SimPy Simulation Function
```python
def run_mmc_simulation(arrival_rate, service_rate, num_servers, sim_time=500):
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=num_servers)
    wait_times, queue_lengths = [], []

    def customer(env, arrival_time):
        with server.request() as req:
            yield req
            wait_times.append(env.now - arrival_time)
            yield env.timeout(np.random.exponential(1.0 / service_rate))

    def generator(env):
        while True:
            yield env.timeout(np.random.exponential(1.0 / arrival_rate))
            queue_lengths.append(len(server.queue))
            env.process(customer(env, env.now))

    env.process(generator(env))
    env.run(until=sim_time)
    return np.mean(wait_times), np.mean(queue_lengths), ...
```

### Step 5: 1000 Simulations
```python
for i in range(1000):
    arr, svc, svrs = sample_random_params()
    results = run_mmc_simulation(arr, svc, svrs, sim_time=500)
    records.append({params + results})
df = pd.DataFrame(records)  # Shape: (1000, 8)
```

### Step 6: ML Pipeline
- **Split**: 80% train, 20% test
- **Scaling**: StandardScaler applied for distance/gradient-based models
- **Models**: 10 models trained and evaluated
- **Evaluation**: RMSE, MAE, R², 5-Fold CV R²

---

## Results Table

### ML Model Performance Comparison

| Rank | Model | RMSE ↓ | MAE ↓ | R² ↑ | CV R² (mean) | CV R² (std) |
|------|-------|--------|-------|------|--------------|-------------|
| 1 | **Random Forest** | **2.1252** | **0.5114** | **0.9895** | 0.9786 | 0.0117 |
| 2 | MLP Neural Net | 2.1713 | 0.7143 | 0.9890 | 0.9807 | 0.0090 |
| 3 | Gradient Boosting | 2.4000 | 0.6179 | 0.9866 | 0.9789 | 0.0105 |
| 4 | AdaBoost | 2.3731 | 0.8892 | 0.9869 | 0.9720 | 0.0157 |
| 5 | Extra Trees | 2.6038 | 0.6246 | 0.9842 | 0.9789 | 0.0093 |
| 6 | Decision Tree | 2.9268 | 0.7459 | 0.9800 | 0.9640 | 0.0241 |
| 7 | KNN | 3.1531 | 0.7275 | 0.9768 | 0.9704 | 0.0138 |
| 8 | SVR | 7.6581 | 1.4315 | 0.8634 | 0.8868 | 0.0505 |
| 9 | Ridge Regression | 9.2780 | 6.8734 | 0.7994 | 0.7986 | 0.0141 |
| 10 | Linear Regression | 9.2847 | 6.9105 | 0.7991 | 0.7985 | 0.0140 |

**Bold** = Best value in each metric column.

### Metric Explanations

- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily. Lower is better.
- **MAE** (Mean Absolute Error): Average of absolute prediction errors. Lower is better. More robust to outliers than RMSE.
- **R²** (Coefficient of Determination): Proportion of variance in wait time explained by the model. 1.0 = perfect, 0.0 = mean-only baseline.
- **CV R²**: 5-fold cross-validated R². Measures generalization — how well the model performs on unseen data folds.

---

## Result Graphs

### Dashboard Overview
![Results Dashboard](results_dashboard.png)

The dashboard contains 9 panels:

| Panel | Description |
|-------|-------------|
| Top-Left | **R² bar chart** — all models ranked. Random Forest leads. |
| Top-Center | **RMSE bar chart** — error magnitude. Linear models worst. |
| Top-Right | **CV R² with error bars** — stability across folds. MLP most consistent. |
| Mid-Left | **Best model (RF) — Actual vs Predicted** — tight scatter near diagonal shows excellent fit |
| Mid-Center | **Worst model (Linear) — Actual vs Predicted** — scatter off diagonal shows poor non-linear capture |
| Mid-Right | **Random Forest feature importances** — `traffic_intensity` dominates |
| Bottom-Left | **Simulation scatter**: Arrival rate vs wait time, colored by servers |
| Bottom-Center | **Traffic intensity vs wait time** — dramatic non-linear rise at ρ → 1 |
| Bottom-Right | **Correlation heatmap** — strong correlation between ρ and wait time |

### Key Observations from Graphs

1. **Non-linearity is visible**: The traffic intensity vs wait time plot shows an exponential curve — linear models cannot capture this
2. **Traffic intensity dominates**: RF feature importance confirms ρ is the most important predictor
3. **Random Forest predictions cluster tightly** on the diagonal — R² = 0.9895
4. **Linear model predictions** fan out significantly, especially at high wait times

---

## Conclusion

### Best Model: Random Forest Regressor

| Why Random Forest wins |
|------------------------|
| Captures **non-linear** queueing dynamics naturally |
| Robust to outliers (high wait times at ρ → 1) |
| No feature scaling required |
| Low variance: CV R² std = 0.0117 (stable across folds) |
| Best test R² = **0.9895** and best MAE = **0.5114** |

### Key Takeaways

1. **Tree-based ensemble models** (RF, GBM, ET) dramatically outperform linear models for this task because queue wait times follow a non-linear relationship with input parameters
2. **Traffic intensity** (ρ) is the most important feature — it encodes the fundamental stability of the queuing system
3. **Simulation + ML** provides a powerful framework: once trained, the ML model can predict outcomes in microseconds rather than running full simulations
4. **Linear models** achieve only R² ≈ 0.80 — they systematically under-predict high wait times near system saturation
5. **MLP Neural Network** achieves the most stable CV performance (std=0.0090), making it a strong alternative to Random Forest

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `SimPy_ML_Assignment.ipynb`
3. Click **Runtime → Run All**
4. All packages install automatically via `!pip install simpy`

### Option 2: Local

```bash
# Clone repository
git clone <your-github-repo-url>
cd <repo-folder>

# Install dependencies
pip install simpy numpy pandas matplotlib seaborn scikit-learn

# Launch notebook
jupyter notebook SimPy_ML_Assignment.ipynb
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| simpy | ≥ 4.0 | Discrete-event simulation |
| numpy | ≥ 1.21 | Numerical computing |
| pandas | ≥ 1.3 | Data management |
| matplotlib | ≥ 3.4 | Plotting |
| seaborn | ≥ 0.11 | Statistical visualization |
| scikit-learn | ≥ 1.0 | Machine learning models |

---

## Files

```
├── SimPy_ML_Assignment.ipynb   # Main Colab notebook
├── simulation_data.csv          # 1000 simulation records
├── ml_model_results.csv         # ML model comparison table
├── results_dashboard.png        # Results visualization
└── README.md                    # This file
```

---

*Assignment: Data Generation using Modelling and Simulation for Machine Learning*
