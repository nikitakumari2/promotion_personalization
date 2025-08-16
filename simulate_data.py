import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker to generate synthetic data
fake = Faker()

# Define the "arms" or actions the model can take
PROMOTIONS = {
    'arm_1': {'type': '10%_off', 'cost': 0.10},
    'arm_2': {'type': '25%_off', 'cost': 0.25},
    'arm_3': {'type': 'free_shipping', 'cost': 0.05}, # Assuming avg shipping cost is 5% of purchase
    'arm_4': {'type': 'buy_one_get_one', 'cost': 0.50}
}

# --- 1. Simulate User Base ---
def create_user_features(num_users=1000):
    """Generates a DataFrame of user features and their hidden preferences."""
    users = []
    for user_id in range(num_users):
        user_profile = {
            'user_id': user_id,
            'days_since_last_purchase': np.random.randint(1, 180),
            'avg_purchase_value': np.random.uniform(20, 300),
            'is_frequent_buyer': np.random.choice([0, 1], p=[0.7, 0.3])
        }
        
        # --- 2. Simulate Hidden User Preferences ---
        # This is the "ground truth" the model needs to learn:
        # For example, frequent buyers might prefer BOGO, while price sensitive users prefer discounts
        prefs = {
            '10%_off_pref': np.random.normal(0.1, 0.05),
            '25%_off_pref': np.random.normal(0.2, 0.1),
            'free_shipping_pref': np.random.normal(0.05, 0.02),
            'buy_one_get_one_pref': np.random.normal(0.3, 0.15)
        }
        
        # Let's say frequent buyers strongly prefer BOGO
        if user_profile['is_frequent_buyer'] == 1:
            prefs['buy_one_get_one_pref'] *= 1.5
            
        # Users with high avg purchase value might prefer free shipping more
        if user_profile['avg_purchase_value'] > 150:
            prefs['free_shipping_pref'] *= 2.0
        
        users.append({**user_profile, **prefs})
        
    return pd.DataFrame(users)

# --- 3. Simulate Conversion ---
def get_reward(user, chosen_arm_type):
    """
    Determines if a user converts based on their preferences.
    Returns 1 for conversion, 0 for no conversion.
    """
    base_conversion_prob = 0.05 # Base likelihood of any user converting
    preference_key = f"{chosen_arm_type}_pref"
    
    # The conversion probability is the base rate plus the user's preference for that offer
    conversion_prob = base_conversion_prob + user[preference_key]
    
    # Ensure probability is between 0 and 1
    conversion_prob = max(0, min(1, conversion_prob))
    
    return np.random.choice([0, 1], p=[1 - conversion_prob, conversion_prob])


if __name__ == '__main__':
    users_df = create_user_features(5000)
    users_df.to_csv('simulated_users.csv', index=False)
    print("Simulated user data created and saved to 'simulated_users.csv'")
    print(users_df.head())