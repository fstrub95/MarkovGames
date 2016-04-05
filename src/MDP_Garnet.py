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

    ################################################################""

    def policy_evaluation_exact_v(self, policy_list, gamma):
        Na = self.a
        Ns = self.s
        Id = np.zeros((self.s, self.s))
        R_pi = np.zeros((self.s))
        Ker_pi = np.zeros((self.s, self.s))
        for i in np.ndindex((self.s)):
            Id[i, i] = 1
        ker_non_stat = np.copy(Id)
        req_non_stat = np.zeros((self.s))
        for policy in policy_list:
            for i in range(self.s):
                R_pi[i] = np.vdot(self.reward[i, :], policy[i, :])
            for i, j in np.ndindex((self.s, self.s)):
                Ker_pi[j, i] = np.vdot(self.kernel[i, j, :], policy[j, :])
            ker_non_stat = gamma * Ker_pi.dot(ker_non_stat)
            req_non_stat = R_pi + gamma * Ker_pi.dot(req_non_stat)
        v_non_stat = np.linalg.solve(Id - ker_non_stat, req_non_stat)
        return v_non_stat

    def policy_evaluation_exact_Q(self, policy_list, gamma):
        Q_non_stat = np.zeros((self.s, self.a))
        v_non_stat = self.policy_evaluation_exact_v(policy_list, gamma)
        for i, j in np.ndindex((self.s, self.a)):
            Q_non_stat[i, j] = self.reward[i, j] + gamma * np.vdot(v_non_stat, self.kernel[:, i, j])
        return Q_non_stat

    def Apply_bellman(self, policy_list, Q_function, gamma):
        Q_function_ = np.copy(Q_function)
        for policy in policy_list:
            ## calcul de la qfunction suivante
            v=np.asarray([np.vdot(Q_function_[j, :], policy[j, :]) for j in range(self.s)])
            for i, j in np.ndindex((self.s, self.a)):
                Q_function_[i, j] = self.reward[i, j] + gamma * np.vdot(v, self.kernel[:, i, j])
        return Q_function_

    def policy_iteration_exact(self, gamma):
        policy = np.zeros((self.s, self.a))
        policy_ = np.zeros((self.s, self.a))
        for i in range(self.s):
            policy[i, 0] = 1
        while not(np.array_equal(policy, policy_)) and not(np.linalg.norm(self.policy_evaluation_exact_v([policy], gamma)-self.policy_evaluation_exact_v([policy_], gamma)) < pow(10, -9)):
            policy_ = np.copy(policy)
            q = self.policy_evaluation_exact_Q([policy_], gamma)
            policy = self.greedy_policy(q)
        return [policy]

    def greedy_policy(self,Q_function):
        policy = np.zeros(((Q_function[:, 0]).size, (Q_function[0, :]).size))
        for i in range((Q_function[:, 0]).size):
            policy[i, np.argmax(Q_function[i, :], axis=0)] = 1
        return policy

    def getQStar(self, gamma):
        return self.policy_evaluation_exact_Q(self.policy_iteration_exact(gamma), gamma)


    def getQpi(self, Q, gamma):
        policy = self.greedy_policy(Q)
        return self.policy_evaluation_exact_Q([policy], gamma)


    def l2(self, estimate, target):
        return np.linalg.norm(estimate - target)/np.linalg.norm(target)

    def l2errorDiffQstarQpi(self, Q, gamma):
        Q_pi   = self.getQpi(Q, gamma)
        Q_star = self.getQStar(gamma)
        return self.l2(Q_pi, Q_star)

    def l2errorBellmanResidual(self, Q, gamma):
        policy = self.greedy_policy(Q)
        TQ = self.Apply_bellman([policy],Q,gamma)
        return self.l2(TQ, Q)