import numpy as np
from collections import defaultdict
def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
#     delta = 0
#     V = np.zeros(env.nS)
#     i = 0
#     while delta < theta:
#         print(i)
#         for s in range(env.nS):
#             if not (s == 0 or s == 15):
#                 v = V[s]
#                 sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
#                 rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
#                 p = np.array([env.P[s][a][0][0] for a in range(0,4)])
#                 pi = np.array(policy[s][:])
#                 V[s] = np.dot(pi.T,p*(rs + discount_factor * V[sp]))
#                 delta = max(delta, np.abs(v-V[s]))
#         i = i+1

    V = np.zeros(env.nS)
    v = np.zeros(env.nS)
    i = 0
    while True:
        delta = 0
#         print("This is iter ", str(i))
        v = np.copy(V)
        for s in range(env.nS):
            if not (s == 0 or s == 15):
                sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
                rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
                p = np.array([env.P[s][a][0][0] for a in range(0,4)])
                pi = np.array(policy[s][:])
                V[s] = np.dot(pi.T,p*(rs + discount_factor * v[sp]))
                delta = max(delta, np.abs(v[s]-V[s]))
#         print("The delta is ", delta)
        if delta < theta:
            break
                
        i = i+1
        
    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    stable = True
    
    P = np.array([list(x.values()) for x in env.P.values()], dtype=int)
    P = P.reshape(env.nS, env.nA, 4)

    
    A = np.arange(env.nA)
             
    print(policy.shape)  
    i = 0
    while stable:
        print("This is iter " + str(i))
        print("loopoing")
        stable = True
        old = np.copy(policy)
        V = policy_eval_v(old, env, discount_factor)

        for s in range(env.nS):
            sp = P[s, A, 1]
            rs = P[s, A, 2]
            p = P[s, A, 0]

            actval = p*(rs + discount_factor * V[sp])
            policy[s] = actval
            if np.any(policy[s] != old[s]):
                stable = False
        i+=1

    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    q = np.zeros((env.nS, env.nA))
    delta = 0
    i = 0
    policy = np.zeros((env.nS, env.nA))
    
    while True:
        delta = 0
        print("This is iter ", str(i))
        q = np.copy(Q)
        print("This is the current q", q)
        for s in range(env.nS):
            for a in range(env.nA):
#                 if not (s == 0 or s == 15): # Make sure to pull from the env.P!!!!!!!!!!!!!!!!!!!!!!!  Do I need this?
                #since we a re given an action, each of these will be a single statement
                sp = env.P[s][a][0][1]
                # We need to find the actions for state s'
                qsa = np.max([q[sp][a] for a in range(0,4)])
                rs = env.P[s][a][0][2]
                p = env.P[s][a][0][0]
                Q[s][a] = p * (rs + discount_factor * qsa)
                delta = max(delta, np.abs(q[s][a]-Q[s][a]))
        print("This is the current Q", Q)
        print("The delta is ", delta)
        if delta < theta:
            break
        i = i+1
    
    for s in range(env.nS):
        mp_loc = np.where(Q[s][:] == np.max(Q[s][:]))
        policy[s][mp_loc[0]] = 1/mp_loc[0].shape[0]

    return policy, Q
def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
#     delta = 0
#     V = np.zeros(env.nS)
#     i = 0
#     while delta < theta:
#         print(i)
#         for s in range(env.nS):
#             if not (s == 0 or s == 15):
#                 v = V[s]
#                 sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
#                 rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
#                 p = np.array([env.P[s][a][0][0] for a in range(0,4)])
#                 pi = np.array(policy[s][:])
#                 V[s] = np.dot(pi.T,p*(rs + discount_factor * V[sp]))
#                 delta = max(delta, np.abs(v-V[s]))
#         i = i+1

    V = np.zeros(env.nS)
    v = np.zeros(env.nS)
    i = 0
    actions = range(0,4)
    while True:
        delta = 0
#         print("This is iter ", str(i))
        v = np.copy(V)
        for s in range(env.nS):
            done_list = [env.P[s][a][0][3] for all a in actions]
            print(done_list)
            if not all(done_list):
                sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
                rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
                p = np.array([env.P[s][a][0][0] for a in range(0,4)])
                pi = np.array(policy[s][:])
                V[s] = np.dot(pi.T,p*(rs + discount_factor * v[sp]))
                delta = max(delta, np.abs(v[s]-V[s]))
#         print("The delta is ", delta)
        if delta < theta:
            break
                
        i = i+1
        
    return np.array(V)
def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
#     delta = 0
#     V = np.zeros(env.nS)
#     i = 0
#     while delta < theta:
#         print(i)
#         for s in range(env.nS):
#             if not (s == 0 or s == 15):
#                 v = V[s]
#                 sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
#                 rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
#                 p = np.array([env.P[s][a][0][0] for a in range(0,4)])
#                 pi = np.array(policy[s][:])
#                 V[s] = np.dot(pi.T,p*(rs + discount_factor * V[sp]))
#                 delta = max(delta, np.abs(v-V[s]))
#         i = i+1

    V = np.zeros(env.nS)
    v = np.zeros(env.nS)
    i = 0
    actions = range(0,4)
    while True:
        delta = 0
#         print("This is iter ", str(i))
        v = np.copy(V)
        for s in range(env.nS):
            done_list = [env.P[s][a][0][3] for all a in actions]
            print(done_list)
            if not all(done_list):
                sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
                rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
                p = np.array([env.P[s][a][0][0] for a in range(0,4)])
                pi = np.array(policy[s][:])
                V[s] = np.dot(pi.T,p*(rs + discount_factor * v[sp]))
                delta = max(delta, np.abs(v[s]-V[s]))
#         print("The delta is ", delta)
        if delta < theta:
            break
                
        i = i+1
        
    return np.array(V)
def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
#     delta = 0
#     V = np.zeros(env.nS)
#     i = 0
#     while delta < theta:
#         print(i)
#         for s in range(env.nS):
#             if not (s == 0 or s == 15):
#                 v = V[s]
#                 sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
#                 rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
#                 p = np.array([env.P[s][a][0][0] for a in range(0,4)])
#                 pi = np.array(policy[s][:])
#                 V[s] = np.dot(pi.T,p*(rs + discount_factor * V[sp]))
#                 delta = max(delta, np.abs(v-V[s]))
#         i = i+1

    V = np.zeros(env.nS)
    v = np.zeros(env.nS)
    i = 0
    actions = range(0,4)
    while True:
        delta = 0
#         print("This is iter ", str(i))
        v = np.copy(V)
        for s in range(env.nS):
            done_list = [env.P[s][a][0][3] for a in actions]
            print(done_list)
            if not all(done_list):
                sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
                rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
                p = np.array([env.P[s][a][0][0] for a in range(0,4)])
                pi = np.array(policy[s][:])
                V[s] = np.dot(pi.T,p*(rs + discount_factor * v[sp]))
                delta = max(delta, np.abs(v[s]-V[s]))
#         print("The delta is ", delta)
        if delta < theta:
            break
                
        i = i+1
        
    return np.array(V)
def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
#     delta = 0
#     V = np.zeros(env.nS)
#     i = 0
#     while delta < theta:
#         print(i)
#         for s in range(env.nS):
#             if not (s == 0 or s == 15):
#                 v = V[s]
#                 sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
#                 rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
#                 p = np.array([env.P[s][a][0][0] for a in range(0,4)])
#                 pi = np.array(policy[s][:])
#                 V[s] = np.dot(pi.T,p*(rs + discount_factor * V[sp]))
#                 delta = max(delta, np.abs(v-V[s]))
#         i = i+1

    V = np.zeros(env.nS)
    v = np.zeros(env.nS)
    i = 0
    actions = range(0,4)
    while True:
        delta = 0
#         print("This is iter ", str(i))
        v = np.copy(V)
        for s in range(env.nS):
            done_list = [env.P[s][a][0][3] for a in actions]
#             print(done_list)
            if not all(done_list):
                sp = np.array([env.P[s][a][0][1] for a in range(0,4)])
                rs = np.array([env.P[s][a][0][2] for a in range(0,4)])
                p = np.array([env.P[s][a][0][0] for a in range(0,4)])
                pi = np.array(policy[s][:])
                V[s] = np.dot(pi.T,p*(rs + discount_factor * v[sp]))
                delta = max(delta, np.abs(v[s]-V[s]))
#         print("The delta is ", delta)
        if delta < theta:
            break
                
        i = i+1
        
    return np.array(V)
