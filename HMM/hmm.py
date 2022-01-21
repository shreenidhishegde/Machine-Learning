
from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
   
    
        for o in range(L):
            o_idx = self.obs_dict.get(Osequence[o])
            for s in range(S):
                if o == 0:
                    alpha[s][o] = self.B[s][o_idx] * self.pi[s]
                else:
                    a_del = 0
                    for s_dash in range(S):
                         a_del += self.A[s_dash][s] * alpha[s_dash][o-1]
                    alpha[s][o] = self.B[s][o_idx] * a_del
        
        return alpha
    
    

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        for o in reversed(range(L)):
            
            for s in range(S):
                beta_sum = 0
                if o == L-1:
                    beta_sum = 1.0
                else:
                    o_prev_idx = self.obs_dict.get(Osequence[o+1])
                    for s_dash in range(S):
                        beta_sum += self.A[s][s_dash] * self.B[s_dash][o_prev_idx] * beta[s_dash][o+1]
                beta[s][o] = beta_sum
        return beta


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        
        prob = 0

        S = len(self.pi)
        L = len(Osequence)
        alpha = self.forward(Osequence)
        for i in range(S):
            prob += alpha[i][L-1]
        return prob


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)

        for s in range(S):
            for l in range(L):
                prob[s][l] = (alpha[s][l]*beta[s][l])/seq_prob
        
        return prob


    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
       
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        
        seq_prob = self.sequence_prob(Osequence)

        for s in range(S):
            for s_dash in range(S):
                for l in range(L-1):
                    o_idx = self.obs_dict.get(Osequence[l+1])
                    prob[s][s_dash][l] = (alpha[s][l] * self.A[s][s_dash] * self.B[s_dash][o_idx] * beta[s_dash][l+1])/seq_prob    
                    
        return prob
        


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################

        rev_state_dict = {v: k for k, v in self.state_dict.items()}
        
        S = len(self.pi)
        L = len(Osequence)
        
        delta = np.zeros([S, L])
        delta_t = np.zeros([S, L], dtype = np.int64)
        
        first_obs_idx = self.obs_dict.get(Osequence[0])

        for s in self.state_dict.values():
            delta[s][0] = self.B[s][first_obs_idx] * self.pi[s]

        for o in range(1, L):
            o_idx = self.obs_dict.get(Osequence[o])
            for s in range(S):
                del_sum=[]
                for s_dash in range(S):
                    del_sum.append(self.A[s_dash][s] * delta[s_dash][o-1])

                delta[s][o] = self.B[s][o_idx] * np.max(del_sum)
                delta_t[s][o-1] = np.argmax(del_sum)

        d_trans = np.array(delta).T
        state_T = np.argmax(d_trans[L-1])
        delta_t[state_T][L-1] = state_T
        
        for i in reversed(range(L)):
            cur_s = delta_t[state_T][i]
            path.append(rev_state_dict.get(cur_s))
            state_T = cur_s
        path.reverse()
        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
