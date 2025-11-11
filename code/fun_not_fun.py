import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from itertools import product

import numpy as np
from itertools import product

# Array di input con valori tra 0 e 1 con passo 0.1
array = np.linspace(0, 1, 10)

# Numero di dimensioni per il prodotto cartesiano (ad esempio, 10)
num_dimensions = 5

# Genera tutte le possibili combinazioni
combinazioni = np.array(list(product(array, repeat=num_dimensions)))

# Mostra il risultato
print("Numero totale di combinazioni:", combinazioni.shape[0])
print("Prima combinazione:", combinazioni[0])
print("Ultima combinazione:", combinazioni[-1])


bounds = [(0, 1)] * 5
def ultra_utility_function(omega, cumulative_returns, r_f, A):
    rf = np.exp(r_f)
    T = len(omega)
    portfolio_returns = 1
    for o, ret in zip(omega, cumulative_returns):
        portfolio_returns *= ((1 - o) * rf) + (o * np.exp(r_f + ret))
    return -np.mean((portfolio_returns ** (T - T * A))) / (1 - A) ** T

def evaluate_combination(omega, cumulative_returns, r_f, A):
    # Calcola il valore della funzione per una singola combinazione
    return ultra_utility_function(omega, cumulative_returns, r_f, A)

resu = Parallel(n_jobs=-1, backend="loky")(
    delayed(evaluate_combination)(
        omega, list(reversed(cumulative_returns_no_uncertainty)), r_f, A,
    ) for omega in combinazioni
)

best_index = np.argmin(resu)
best_omega = combinazioni[best_index]

# Visualizzazione del risultato
print(best_omega)



def ultra_utility_function(omega,cumulative_returns, r_f, A):
    
    rf=np.exp(r_f)
    T=len(omega)
    
    portfolio_returns = 1
    for o,ret in zip(omega,cumulative_returns):
        portfolio_returns *= ((1 - o) * rf) + (o * np.exp(r_f + ret))
    return -np.mean((portfolio_returns ** (T-T*A)))/(1-A)**T  # Minimize negative utility

resu = []
bounds = [(0, 1)] * 20
for omega in combinazioni:
    resu.append(minimize(ultra_utility_function,x0=omega,args=(list(reversed(cumulative_returns_no_uncertainty)),r_f,20),bounds=bounds,method="L-BFGS-B").x)

for A in Aversion:
    print(A)
    W_k = []
    W_k_u = []
    
    W = np.ones([n_samples])
    W_u = np.ones(n_samples)

    W_k = [np.ones(W.shape) for _ in range(K)]         
    W_k_u = [np.ones(W_u.shape) for _ in range(K)]    
    for T_prime in range(K):
            for i,cumulative_returns_w in enumerate(cumulative_returns_with_uncertainty[-T_prime]):
                result = minimize(
                    utility_function,
                    x0=0.5,
                    args=(W_k_u[T_prime],cumulative_returns_w, r_f, A),
                    bounds=[(0, 1)],
                    method="Powell"
                )

                results_predictability_uncertainty[T_prime][0][A].append(result.x[0])
                W_k_u[T_prime] =  W_k_u[T_prime-1]*((1 - result.x[0]) * np.exp(r_f*(T/K)) + (result.x[0] * np.exp(r_f*(T/K) + cumulative_returns_with_uncertainty[T_prime])))
            
            for i,cumulative_returns_nw in enumerate(cumulative_returns_no_uncertainty[-T_prime]):
                result = minimize(
                    utility_function,
                    x0=0.5,
                    args=(W_k[T_prime],cumulative_returns_nw, r_f, A),
                    bounds=[(0, 1)],
                    method="Powell"
                )
                results_predictability_no_uncertainty[T_prime][0][A].append(result.x[0])
                W_k[T_prime] =  W_k[T_prime-1]*((1 - result.x[0]) * np.exp(r_f*(T/K)) + (result.x[0] * np.exp(r_f*(T/K) + cumulative_returns_no_uncertainty[T_prime])))







X = res.params
test = np.matrix(X)
Aversion = [5, 10, 20]
K = 20
T = 20
n_samples = 100000
r_f = np.mean(returns_bond)


def utility_function(omega,W,cumulative_returns, r_f, A):
    portfolio_returns = W*((1 - omega) * np.exp(r_f*(T/K))) + (omega * np.exp(r_f*(T/K) + cumulative_returns))
    return -np.mean(((portfolio_returns) ** (1 - A)))/ (1 - A)  # Minimize negative utility

n_samples = 100000


div_values = [[0,-0.08],[0,-0.0008],[0,np.mean(First_Diff_Yield)],[0,0.0008],[0,0.004]]

forecast = []
for d in div_values:
    d[0] = Stocks_Model.iloc[-1].values[0]
    resu_unc = res.forecast(np.array([d]), K)  # For uncertainty
    forecast.append(resu_unc)

z_T_f = div_values  # Current state

results_predictability_un = []
results_predictability_no = []
for i in range(len(div_values)):
    results_predictability_un.append({A: [] for A in Aversion})
    results_predictability_no.append({A: [] for A in Aversion})

results_predictability_uncertainty = [results_predictability_un] *K
results_predictability_no_uncertainty = [results_predictability_no] * K


cumulative_returns_with_uncertainty = np.zeros((len(div_values), n_samples))
cumulative_returns = np.zeros((len(div_values), n_samples))
cumulative_returns_no_uncertainty = [np.zeros(cumulative_returns.shape) for _ in range(K)]

"""
dividend_yield_range = list(np.linspace(
    np.mean(First_Diff_Yield) - 3 * np.std(First_Diff_Yield),
    np.mean(First_Diff_Yield) + 3 * np.std(First_Diff_Yield),
    20
))
"""
W = np.ones([len(div_values),n_samples])
W_k = [np.ones(W.shape) for _ in range(K+1)]

for T_prime in range(K):  
        for i, z_T in zip(range(len(div_values)), z_T_f):
        
            mean_pred, cov_pred = compute_mean_and_variance_recursive(
                z_T,
                np.squeeze(np.asarray(test[0])),
                test[1:, :],
                res.resid_acov()[0],
                T_prime+1
            )
            mean_predictive_returns = multivariate_normal.rvs(mean=mean_pred, cov=cov_pred, size=n_samples)
            cumulative_returns_no_uncertainty[T_prime][i] = cumulative_returns_no_uncertainty[T_prime][i]+mean_predictive_returns[:, 0]
for A in Aversion:
    W_k = []
    W = np.ones([len(div_values),n_samples])
    W_k = [np.ones(W.shape) for _ in range(K+1)]                  
    for T_prime in range(K):  
            for i,cumulative_returns in enumerate(cumulative_returns_no_uncertainty[-T_prime]):
                result = minimize(
                    utility_function,
                    x0=0.5,
                    args=(W_k[T_prime][i],cumulative_returns, r_f, A),
                    bounds=[(0, 1)],
                    method="L-BFGS-B"
                )
                results_predictability_no_uncertainty[T_prime][i][A].append(result.x[0])
                W_k[T_prime+1][i] =  W_k[T_prime][i]*((1 - result.x[0]) * np.exp(r_f*(T/K)) + (result.x[0] * np.exp(r_f*(T/K) + cumulative_returns_no_uncertainty[T_prime][i])))



plot_titles = ["A=5", "A=10", "A=20"]
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
for T_prime in range(2,11):
    for idx, A in enumerate(Aversion):
            ax = axes[idx, 1]
            for info in results_predictability_no_uncertainty[T_prime]:
                ax.plot(range(2, 11), info[A][2:11], label="Predictability (No Uncertainty)", linestyle="-.")
            ax.set_title(plot_titles[idx])
            ax.set_xlabel("Investment Horizon (Years)")
            ax.set_ylabel("Optimal Allocation to Stocks")
            ax.grid()

    plt.tight_layout()
    plt.show()