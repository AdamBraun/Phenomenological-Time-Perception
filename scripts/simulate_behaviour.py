import numpy as np
import pandas as pd

def simulate_behaviour(n_participants=30, n_trials=192, epsilon=0.1, delta=0.3):
    """
    Simulate 2AFC task data for N participants, with 4x4 displacement-duration grid.
    Returns a DataFrame with binary choices and response times.
    """
    displacements = [0.05, 0.10, 0.20, 0.40]
    durations = [0.2, 0.4, 0.8, 1.6]
    trials_per_condition = n_trials // (len(displacements) * len(durations))
    
    data = []
    for pid in range(n_participants):
        for disp in displacements:
            for dur in durations:
                for _ in range(trials_per_condition):
                    # Simulate perceived ticks based on ε-δ model
                    k_eps = epsilon * (0.10 / disp)
                    k_delta = delta * (0.10 / disp)
                    ticks = 0 if dur < k_eps else 1 if dur <= k_delta else np.ceil(dur / k_delta)
                    # Simulate binary choice (logistic probability based on ticks)
                    prob_correct = 1 / (1 + np.exp(-ticks))  # Sigmoid function
                    choice = np.random.binomial(1, prob_correct)
                    # Simulate response time (normal, centered at 500 ms)
                    rt = np.random.normal(500, 100)
                    while rt < 200 or rt > 5000:  # Enforce RT bounds
                        rt = np.random.normal(500, 100)
                    data.append([pid, disp, dur, choice, rt])
    
    # Catch trials (5 per participant, obvious differences)
    for pid in range(n_participants):
        for _ in range(5):
            disp = np.random.choice([0.40, 0.80])  # Obvious displacement
            dur = np.random.choice([1.6, 3.2])     # Obvious duration
            choice = np.random.binomial(1, 0.95)   # High accuracy
            rt = np.random.normal(400, 50)
            while rt < 200 or rt > 5000:
                rt = np.random.normal(400, 50)
            data.append([pid, disp, dur, choice, rt])
    
    df = pd.DataFrame(data, columns=['participant_id', 'displacement', 'duration', 'choice', 'response_time'])
    df.to_csv('simulated_2afc_data.csv', index=False)
    return df

if __name__ == "__main__":
    simulate_behaviour()
    print("Simulated data saved to 'simulated_2afc_data.csv'")