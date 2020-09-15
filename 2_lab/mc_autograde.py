import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        probs = []
        for state, action in zip(states, actions):
            policyaction = self.sample_action(state)
            probs.append(float(policyaction == action))
        
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        if state[0] == 20 or state[0] == 21:
            return 0
        return 1

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    states.append(env.reset())
    t = 0
    while True:
        a = policy.sample_action(states[t])
        sp, r, done, _ = env.step(a)
        
        actions.append(a)
        rewards.append(r)
        dones.append(done)
        if done:
            break
        
        states.append(sp)
        t += 1
    
    return states, actions, rewards, dones

def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                S.append(s)
                returns[s].append(G)
                V[s] = sum(returns[s])/len(returns[s])
                
    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        
        return np.zeros(len(states)) + 0.5
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        return np.random.choice(a=[0, 1])

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                S.append(s)
                returns[s].append(G)
                V[s] += pt[idx]/pb[idx] * (G - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                S.append(s)
                returns[s].append(G)
                V[s] += pt[idx]/pb[idx] * (G/len(returns) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                V[s] += pt[idx]/pb[idx] * (G/returns_count[s] - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                V[s] = V[s] + pt[idx]/pb[idx] * (sum(returns)/returns_count[s] - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                V[s] = V[s] + pt[idx]/pb[idx] * (G/returns_count[s] - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(3)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                print(pt[idx], pb[idx])
                
                V[s] = V[s] + pt[idx]/pb[idx] * (G/returns_count[s] - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(3)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                print(pt[idx], pb[idx])
                
                V[s] = V[s] + pt[idx]/pb[idx] * (rewards[idx] - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(3)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                print(pt[idx], pb[idx])
                
                V[s] = V[s] + pt[idx]/pb[idx] * (G - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(3)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                print(pt[idx], pb[idx])
                
                V[s] = V[s] + pt[idx]/pb[idx] * (G/idx - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(3)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                print(pt[idx], pb[idx])
                
                V[s] = V[s] + pt[idx]/pb[idx] * (G/(idx+1) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
                print(pt[idx], pb[idx])
                
                V[s] = V[s] + pt[idx]/pb[idx] * (G/(idx+1) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        G = 0
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + pt[idx]/pb[idx] * (G/(idx+1) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (G - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (sum(returns[s])/len(returns) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    S = defaultdict(list)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        S = []
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S[s].append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (sum(returns[s])/len(S[s]) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    S = defaultdict(list)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S[s].append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (sum(returns[s])/len(S[s]) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    S = defaultdict(list)
    
    returns = defaultdict(list)
    
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
            
            print(W)
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S[s].append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (sum(returns[s])/len(S[s]) - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
            
            print(W)
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S[s].append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (G - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
            
            print(W)
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (G - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
                  
            print("W", W)
            print("G", G)
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (G - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            W = pt[idx]/pb[idx]
                  
            print("W", W)
            print("G", G)
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (G - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            W = pt[idx]/pb[idx]
                  
            print("W", W)
            print("G", G)
            
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(G)
#                 print(pt[idx], pb[idx])
                
                V[s] = V[s] + W * (G/returns_count[s] - V[s]) 
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            W = pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(W*G)                
                V[s] = sum(returns[s])/len(states)
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                S.append(s)
                returns[s].append(W*G)                
                V[s] = sum(returns[s])/len(states)
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

        # Keeps track of current V and count of returns for each state
    # to calculate an update.

    # nvm ignore this
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                S.append(s)                
                V[s] = V[s] + W * (G  - V[s]) 
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                S.append(s)                
                V[s] = V[s] + W * (G  - V[s]) 
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(lambda: float(1))
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns[s].append()
                S.append(s)                
                V[s] = V[s] + W * (G  - V[s]) 
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns[s].append(W*G)
                S.append(s)                
                V[s] = sum(returns[s])/len(rewards) 
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns[s].append(W*G)
                S.append(s)                
                V[s] = sum(returns[s])/len(returns[s]) 
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returnscount[s] += 1
                returns[s].append(G)
                S.append(s)                
                V[s] += 1/returnscount[s] * (W * G - V[s])
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                returns[s].append(G)
                S.append(s)                
                V[s] += 1/returnscount[s] * (W * G - V[s])
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(5)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                returns[s].append(G)
                S.append(s)                
                V[s] += 1/returns_count[s] * (W * G - V[s])
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(1000)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                returns[s].append(G)
                S.append(s)                
                V[s] = V[s] + (1/returns_count[s]) * (W*G - V[s])
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(1000)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                returns[s].append(G)
                S.append(s)         
                
                print("update rate", 1/returns_count[s])
                print("target", W*G)
                
                V[s] = V[s] + (1/returns_count[s]) * (W*G - V[s])
                
    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    
    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    
    returns = defaultdict(list)
    returns_count = defaultdict(float)
    
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        
        pt = target_policy.get_probs(states, actions)
        pb = behavior_policy.get_probs(states, actions)
        
        W = 1
        G = 0
        
        for idx in range(len(states) - 1, -1, -1):
            S = []
            G = discount_factor * G + rewards[idx]
            
            W *= pt[idx]/pb[idx]
                  
            s = states[idx]
            if s not in S:
                returns_count[s] += 1
                returns[s].append(G)
                S.append(s)         
                
                V[s] = V[s] + (1/returns_count[s]) * (W*G - V[s])
                
    return V
