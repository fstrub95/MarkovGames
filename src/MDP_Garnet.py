__author__ = 'julien-perolat'
import sys
import numpy as np
import random as rd
from scipy import *
from tools import *



class Garnet_MDP:
    def __init__(self, Ns, Na, Nb, sparsity, neighbor):
        # nbr etats
        self.s = Ns
        # nbr actions
        self.a = Na
        # kernel
        kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
        for i, j in np.ndindex((Ns, Na)):
            echantillon = rd.sample(list(set(range(Ns)).intersection(range(i-neighbor,i+neighbor))), Nb)
            cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
            for k in range(Nb):
                kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]
        self.kernel = kernel
        # reward
        reward = np.random.randn(Ns, Na)
        masque_reward = np.zeros((Ns, Na))
        N_sparsity = int(Na * Ns * sparsity)
        i = 0
        while i < N_sparsity:
            i_ = rd.randint(0, Ns - 1)
            j_ = rd.randint(0, Na - 1)
            if masque_reward[i_, j_] == 0:
                masque_reward[i_, j_] = 1
                i += 1
        reward = reward * masque_reward
        self.reward = reward

    def sample_state_unif(self):
        return rd.randint(0, self.s - 1)

    def action_set_list(self, s = None):
        return range(self.a)

    def next_state(self, s, action_list):
        return random_distr(zip(range(self.s), list(self.kernel[:, s, action_list[0]])))

    def rollout(self, s, action_list, m, policy_vect):
        # policy_vect is a np.array((Ns,Na))
        state_action = [s, action_list]
        # print state_action
        rollout = []
        for i in range(m):
            #print (-i-1) % len(policy_vect)
            rollout.append([state_action[0], state_action[1], self.reward[state_action[0], state_action[1][0]]])
            next_s = self.next_state(state_action[0], state_action[1])
            state_action = [next_s, sample_action([self.action_set_list(next_s)],
                                                  [list((policy_vect[(-i-1) % len(policy_vect)])[next_s, :])])]
            #print (i % len(policy_vect))
            #print (policy_vect[i % len(policy_vect)])[next_s, :]
            #print state_action
        # print rollout[0]
        return rollout

    def uniform_batch_data(self,N, n_resample = 1):
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        ech = []
        for j in range(N):
            state = self.sample_state_unif()
            action_list = self.action_set_list(state)
            action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
            # print action
            for k in range(n_resample):
                ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
        sars_ = [(s, a, r, s_) for [[s, [a], r], [s_, [a_], r_]] in ech]
        return sars_
