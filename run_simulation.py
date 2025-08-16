import pandas as pd
import numpy as np
from bandit import LinUCB
from simulate_data import PROMOTIONS, get_reward

# --- 1. Load Data and Define Features ---
users_df = pd.read_csv('simulated_users.csv')
# Define which columns are user features (the "context")
# added a bias/intercept term for the linear model
USER_FEATURES = ['days_since_last_purchase', 'avg_purchase_value', 'is_frequent_buyer']

# --- 2. Simulation Setup ---
N_USERS = len(users_df)
N_ARMS = len(PROMOTIONS)
N_FEATURES = len(USER_FEATURES) + 1  # +1 for the intercept

# Control group gets the standard '10%_off' offer
CONTROL_ARM_TYPE = PROMOTIONS['arm_1']['type']

# Initialize the bandit
bandit = LinUCB(n_arms=N_ARMS, n_features=N_FEATURES, alpha=1.5)

# --- 3. Run Simulation Loop ---
results = []
arm_names = list(PROMOTIONS.keys())
arm_types = [PROMOTIONS[key]['type'] for key in arm_names]

for i in range(N_USERS):
    user_row = users_df.iloc[i]
    
    # --- Control Group ---
    control_reward = get_reward(user_row, CONTROL_ARM_TYPE)
    
    # --- Treatment Group (Bandit) ---
    # Create the context vector, adding an intercept term
    context = np.array([1] + [user_row[feat] for feat in USER_FEATURES])
    
    # 1. Bandit selects an arm
    chosen_arm_idx = bandit.select_arm(context)
    chosen_arm_type = arm_types[chosen_arm_idx]
    
    # 2. Get the reward for the bandit's choice
    bandit_reward = get_reward(user_row, chosen_arm_type)
    
    # 3. Update the bandit with the result
    bandit.update(chosen_arm_idx, context, bandit_reward)
    
    # Store results
    results.append({
        'user_id': user_row['user_id'],
        'control_reward': control_reward,
        'bandit_reward': bandit_reward,
        'bandit_arm': chosen_arm_type
    })

# --- 4. Analyze Results ---
results_df = pd.DataFrame(results)

# Calculate cumulative rewards
results_df['cumulative_control_reward'] = results_df['control_reward'].cumsum()
results_df['cumulative_bandit_reward'] = results_df['bandit_reward'].cumsum()

# Calculate conversion rates
control_conversion_rate = results_df['control_reward'].mean()
bandit_conversion_rate = results_df['bandit_reward'].mean()
lift = (bandit_conversion_rate - control_conversion_rate) / control_conversion_rate * 100

print("\n--- Simulation Complete ---")
print(f"Control Group Conversion Rate: {control_conversion_rate:.4f}")
print(f"Bandit (Treatment) Group Conversion Rate: {bandit_conversion_rate:.4f}")
print(f"Percentage Lift: {lift:.2f}%")

# results for visualization
results_df.to_csv('simulation_results.csv', index=False)