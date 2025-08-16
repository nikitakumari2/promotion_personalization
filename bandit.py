import numpy as np

class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0):
        """
        Initializes the LinUCB bandit.
        
        Args:
            n_arms (int): Number of arms (promotions).
            n_features (int): Number of user features (context).
            alpha (float): Exploration parameter. Higher alpha means more exploration.
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # Initialize parameters for each arm
        # A: A matrix (d x d) for each arm, initialized as identity matrix
        # b: A vector (d x 1) for each arm, initialized as zeros
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, user_context):
        """
        Selects an arm for a given user context based on the UCB formula
        
        Args:
            user_context (np.array): The feature vector for the current user
            
        Returns:
            int: The index of the chosen arm
        """
        p_t = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Calculate the inverse of A for the current arm
            A_inv = np.linalg.inv(self.A[arm])
            
            # Estimate the coefficients (theta) for the linear model of the arm
            theta = A_inv @ self.b[arm]
            
            # Calculate the predicted reward (p_t) for the arm
            # This consists of two parts:
            # 1. Exploitation: theta.T @ user_context (predicted reward)
            # 2. Exploration: alpha * sqrt(user_context.T @ A_inv @ user_context) (uncertainty bonus)
            exploitation_term = theta.T @ user_context
            exploration_term = self.alpha * np.sqrt(user_context.T @ A_inv @ user_context)
            
            p_t[arm] = exploitation_term + exploration_term
            
        # Choose the arm with the highest UCB score
        chosen_arm = np.argmax(p_t)
        return chosen_arm

    def update(self, chosen_arm, user_context, reward):
        """        
        Args:
            chosen_arm (int): The arm that was chosen.
            user_context (np.array): The user's feature vector
            reward (int): The observed reward (0 or 1)
        """
        self.A[chosen_arm] += np.outer(user_context, user_context)
        self.b[chosen_arm] += reward * user_context