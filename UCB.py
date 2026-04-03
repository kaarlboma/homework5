import numpy as np

def UCB(delta, Mu, n, num_simulations):
    K = len(Mu)
    mu_star = np.max(Mu)


    regrets = np.zeros(num_simulations)

    for sim in range(num_simulations):
        counts = np.zeros(K)
        means = np.zeros(K)
        total_regret = 0

        for i in range(K):
            reward = np.random.normal(Mu[i], 1)
            counts[i] += 1
            means[i] = reward
        
            total_regret += mu_star - Mu[i]
        
        for t in range(K, n):
            ucb_values = means + (np.sqrt((2 * np.log((2 * n * K) / delta)) / counts))
            arm = np.argmax(ucb_values)
            reward = np.random.normal(Mu[arm], 1)
            counts[arm] += 1
            means[arm] += (reward - means[arm]) / counts[arm]
            
            # Compute Regret
            total_regret += mu_star - Mu[arm]

        regrets[sim] = total_regret

    # Calculate average regret + stderr
    average_regret = np.average(regrets)

    return average_regret