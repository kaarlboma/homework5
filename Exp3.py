import numpy as np

def Exp3(Mu, eta, n, num_simulations):
    K = len(Mu)
    mu_star = np.max(Mu)
    regrets = []

    for sim in range(num_simulations):
        rewards = np.random.binomial(1, Mu, size=(n, K))
        regret = []
        total_reward = 0
        log_w = np.zeros(K)
        for t in range(n):
            log_w_shifted = log_w - np.max(log_w)
            w = np.exp(log_w_shifted)  
            p = w / np.sum(w)
            a_t = np.random.choice(K, p = p)
            r_t = rewards[t, a_t]
            total_reward += r_t
            r_tilde = np.zeros(K)
            r_tilde[a_t] = r_t / p[a_t]
            log_w += eta * r_tilde
            regret.append((t + 1) * mu_star - total_reward)
    
        regrets.append(regret[-1])
    avg_regret = np.mean(regrets)
    return avg_regret