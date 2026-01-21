import numpy as np
from scipy.special import softmax

from utils.policy import BasePolicy


class OBPPolicyAdapter(BasePolicy):
    """
    Adapter to wrap Open Bandit Pipeline (OBP) policies to match the BasePolicy interface.

    This adapter handles both context-free and contextual policies from OBP, converting
    their outputs to probability distributions as required by the cross-validated OPE framework.

    Parameters
    ----------
    obp_policy : obp.policy.BasePolicy
        The OBP policy instance to wrap
    a_num : int
        Number of actions
    policy_type : str, default="contextual"
        Type of policy: "contextual" or "context_free"
    temperature : float, default=1.0
        Temperature parameter for softmax conversion of scores to probabilities.
        Only used for contextual policies.
    """

    def __init__(self, obp_policy, a_num, policy_type="contextual", temperature=1.0, **kwargs):
        super().__init__(a_num=a_num)
        self.obp_policy = obp_policy
        self.policy_type = policy_type
        self.temperature = temperature
        self.n_actions = a_num

    def fit(self, X, y, **kwargs):
        """
        Train the policy on the given data.

        For OBP contextual policies, this simulates online learning by iterating
        through the training data and calling select_action() + update_params().

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Context features
        y : ndarray, shape (n_samples,)
            Action labels (ground truth actions)

        Returns
        -------
        self : OBPPolicyAdapter
        """
        # For context-free policies that don't need training, do nothing
        if self.policy_type == "context_free":
            return self

        # For contextual policies, simulate online learning
        if hasattr(self.obp_policy, 'select_action'):
            for i in range(len(X)):
                context = X[i:i+1]  # Keep as 2D array with shape (1, n_features)

                # Select action based on current policy
                action = self.obp_policy.select_action(context)

                # Convert action to integer if it's an array
                if isinstance(action, np.ndarray):
                    action = int(action[0])

                # Compute reward (1 if action matches ground truth, 0 otherwise)
                reward = float(action == y[i])

                # Update policy parameters if the policy supports it
                if hasattr(self.obp_policy, 'update_params'):
                    self.obp_policy.update_params(
                        action=action,
                        reward=reward,
                        context=context
                    )

        return self

    def __call__(self, X, a=None):
        """
        Compute action probabilities for the given contexts.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Context features
        a : int or ndarray, optional
            Specific action(s) to get probabilities for. If None, returns
            probabilities for all actions.

        Returns
        -------
        probabilities : ndarray
            If a is None: shape (n_samples, n_actions) - full probability distribution
            If a is not None: shape (n_samples,) - probabilities for specified actions
        """
        n_samples = len(X)

        # Handle context-free policies that provide batch action distributions
        if self.policy_type == "context_free" and hasattr(self.obp_policy, 'compute_batch_action_dist'):
            # Check if the policy supports n_sim parameter (e.g., EpsilonGreedy)
            # or just n_rounds (e.g., Random)
            try:
                action_dist = self.obp_policy.compute_batch_action_dist(
                    n_rounds=n_samples,
                    n_sim=10000
                )
            except TypeError:
                # Fallback for policies that don't support n_sim
                action_dist = self.obp_policy.compute_batch_action_dist(
                    n_rounds=n_samples
                )

            # Ensure consistent shape: (n_samples, n_actions)
            if action_dist.ndim == 3:
                action_dist = action_dist.squeeze(-1)
        else:
            # For contextual policies, compute scores and convert to probabilities
            action_dist = self._compute_contextual_distribution(X)

        # Return full distribution or specific action probabilities
        if a is None:
            return action_dist
        else:
            return action_dist[np.arange(n_samples), a]

    def _compute_contextual_distribution(self, X):
        """
        Compute probability distribution for contextual policies.

        Since OBP contextual policies typically only provide select_action() which
        returns a single action, we need to compute scores for all actions and
        convert them to probabilities using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Context features

        Returns
        -------
        action_dist : ndarray, shape (n_samples, n_actions)
            Probability distribution over actions
        """
        n_samples = len(X)
        scores = np.zeros((n_samples, self.n_actions))

        # Compute scores for each context-action pair
        for i in range(len(X)):
            context_2d = X[i:i+1]  # Keep as 2D array with shape (1, n_features)

            for action in range(self.n_actions):
                scores[i, action] = self._get_action_score(context_2d, action)

        # Convert scores to probabilities using softmax
        action_dist = softmax(scores / self.temperature, axis=1)
        return action_dist

    def _get_action_score(self, context, action):
        """
        Get the score for a specific action given a context.

        This method accesses the internal parameters of OBP policies to compute
        scores that represent the policy's preference for each action.

        Parameters
        ----------
        context : ndarray, shape (1, n_features)
            Context features (single sample)
        action : int
            Action index

        Returns
        -------
        score : float
            Score for the given action (higher is better)
        """
        policy_name = self.obp_policy.__class__.__name__

        # For Linear UCB and Linear Thompson Sampling policies
        if hasattr(self.obp_policy, 'theta_hat') and hasattr(self.obp_policy, 'A_inv'):
            # LinUCB: score = theta^T * x + alpha * sqrt(x^T * A_inv * x)
            # LinTS: score = theta^T * x (theta is sampled)
            # theta_hat shape: (dim, n_actions), so theta_hat[:, action] gives parameters for this action
            theta = self.obp_policy.theta_hat[:, action]  # Shape: (dim,)
            expected_reward = np.dot(context, theta)  # Shape: (1,)

            # Add exploration bonus for UCB
            if 'UCB' in policy_name or 'LinUCB' in policy_name:
                A_inv = self.obp_policy.A_inv[action]  # Shape: (dim, dim)
                alpha = getattr(self.obp_policy, 'epsilon', 1.0)  # Note: OBP uses 'epsilon' not 'alpha'
                exploration_bonus = alpha * np.sqrt(np.dot(np.dot(context, A_inv), context.T))
                return expected_reward[0] + exploration_bonus[0, 0]

            return expected_reward[0]

        # For Logistic policies
        elif hasattr(self.obp_policy, 'alpha_hat') and hasattr(self.obp_policy, 'beta_hat'):
            # Logistic models: score based on logistic function parameters
            alpha = self.obp_policy.alpha_hat[action]
            beta = self.obp_policy.beta_hat[action]
            expected_reward = alpha + np.dot(context, beta)

            # Add exploration for Logistic UCB
            if 'UCB' in policy_name:
                # Simplified exploration bonus
                exploration_bonus = getattr(self.obp_policy, 'epsilon', 0.1)
                return expected_reward[0] + exploration_bonus

            return expected_reward[0]

        # For epsilon-greedy and other policies without explicit scoring
        elif hasattr(self.obp_policy, 'base_policy'):
            # Epsilon-greedy wraps another policy - recurse
            base_adapter = OBPPolicyAdapter(
                self.obp_policy.base_policy,
                self.n_actions,
                policy_type="contextual",
                temperature=self.temperature
            )
            return base_adapter._get_action_score(context, action)

        # For simple policies without scoring, use uniform distribution
        else:
            return 0.0
