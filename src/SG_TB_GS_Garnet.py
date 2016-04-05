__author__ = 'julien-perolat'


import sys
import numpy as np
import random as rd
from scipy import *
from tools import *
from itertools import product


rd.seed()

BETA1 = 0.01
BETA2 = 0.99
####################################


class Garnet_SG_TB_GS:
    def __init__(self, Ns, Na, Nb, sparsity, type_gar):
        if type_gar == "SA_T2":
            Ns, Na, kernel, reward0, reward1, control = garnet_gen_sa( Ns, Na, Nb, sparsity, Nb)
        elif type_gar == "SA_T1":
            Ns, Na, kernel, reward0, reward1, control = garnet_gen_sa( Ns, Na, Nb, sparsity, Ns)
        elif type_gar == "S_T2":
            Ns, Na, kernel, reward0, reward1, control = garnet_gen_s( Ns, Na, Nb, sparsity, Nb)
        elif type_gar == "S_T1":
            Ns, Na, kernel, reward0, reward1, control = garnet_gen_s( Ns, Na, Nb, sparsity, Ns)
        elif type_gar == "S_linear_T2":
            Ns, Na, kernel, reward0, reward1, control = garnet_gen_s_linear( Ns, Na, Nb, sparsity, Nb)
        elif type_gar == "S_linear_T1":
            Ns, Na, kernel, reward0, reward1, control = garnet_gen_s_linear( Ns, Na, Nb, sparsity, Ns)

        # nbr etats
        self.s = Ns
        # nbr actions
        self.a = Na
        # kernel
        self.kernel = kernel
        # reward
        self.reward0 = reward0
        self.reward1 = reward1
        self.control = control ### 1 si maximise 0 si minimise


    def sample_state_unif(self):
        return rd.randint(0, self.s - 1)

    def action_set_list(self, s):
        return range(self.a)

    def next_state(self, s, action_list):
        #print action_list[0]
        #print list(self.kernel[:, s, action_list[0]])
        return random_distr(zip(range(self.s), list(self.kernel[:, s, action_list[0]])))

    def rollout(self, s, action_list, m, policy_vect):
        # policy_vect is a np.array((Ns,Na))
        state_action = [s, action_list]
        # print state_action
        rollout = []
        for i in range(m):
            rollout.append([state_action[0],
                            state_action[1],
                            [self.reward0[state_action[0], state_action[1][0]], self.reward1[state_action[0], state_action[1][0]]]])
            next_s = self.next_state(state_action[0], state_action[1])
            state_action = [next_s, sample_action([self.action_set_list(next_s)],
                                                  [list((policy_vect[(- i - 1) % len(policy_vect)])[next_s, :])])]
        return rollout

    def uniform_batch_data(self,N, n_resample = 1):
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        ech = []
        for j in range(N):
            state = self.sample_state_unif()
            action_list = self.action_set_list(state)
            action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
            #print action
            for k in range(n_resample):
                ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
        sars_ = [(s, a, self.control[s], r0, r1, s_, self.control[s_]) for [[s, [a], [r0, r1]], [s_, [a_], [r_0, r_1]]] in ech]
        return sars_


    def policy_evaluation_exact_v(self, policy_list, gamma):
        Na = self.a
        Ns = self.s
        Id = np.zeros((self.s, self.s))
        R_pi0 = np.zeros((self.s))
        R_pi1 = np.zeros((self.s))
        Ker_pi = np.zeros((self.s, self.s))
        for i in np.ndindex((self.s)):
            Id[i, i] = 1
        ker_non_stat = np.copy(Id)
        req_non_stat0 = np.zeros((self.s))
        req_non_stat1 = np.zeros((self.s))
        for policy in policy_list:
            for i in range(self.s):
                R_pi0[i] = np.vdot(self.reward0[i, :], policy[i, :])
                R_pi1[i] = np.vdot(self.reward1[i, :], policy[i, :])
            for i, j in np.ndindex((self.s, self.s)):
                Ker_pi[j, i] = np.vdot(self.kernel[i, j, :], policy[j, :])
            ker_non_stat = gamma * Ker_pi.dot(ker_non_stat)
            req_non_stat0 = R_pi0 + gamma * Ker_pi.dot(req_non_stat0)
            req_non_stat1 = R_pi1 + gamma * Ker_pi.dot(req_non_stat1)
        v_non_stat0 = np.linalg.solve(Id - ker_non_stat, req_non_stat0)
        v_non_stat1 = np.linalg.solve(Id - ker_non_stat, req_non_stat1)
        return v_non_stat0, v_non_stat1

    def policy_evaluation_exact_Q(self, policy_list, gamma):
        Q_non_stat0 = np.zeros((self.s, self.a))
        Q_non_stat1 = np.zeros((self.s, self.a))
        v_non_stat0, v_non_stat1 = self.policy_evaluation_exact_v(policy_list, gamma)
        for i, j in np.ndindex((self.s, self.a)):
            Q_non_stat0[i, j] = self.reward0[i, j] + gamma * np.vdot(v_non_stat0, self.kernel[:, i, j])
            Q_non_stat1[i, j] = self.reward1[i, j] + gamma * np.vdot(v_non_stat1, self.kernel[:, i, j])
        return Q_non_stat0, Q_non_stat1

    def greedy_policy(self, Q_function0, Q_function1):
        policy = np.zeros(((Q_function0[:, 0]).size, (Q_function1[0, :]).size))
        for i in range(self.s):
            if self.control[i] == 1:
                policy[i, np.argmax(Q_function1[i, :], axis=0)] = 1
            else:
                policy[i, np.argmax(Q_function0[i, :], axis=0)] = 1
        return policy

    def greedy_best_response(self, policy_list, Q_function0, Q_function1, gamma):
        res_policy_list0 = []
        res_policy_list1 = []
        Q_function_0 = np.copy(Q_function0)
        Q_function_1 = np.copy(Q_function1)
        for policy in policy_list:
            policy_0 = self.greedy_policy(Q_function_0, Q_function_1)
            policy_1 = self.greedy_policy(Q_function_0, Q_function_1)
            for i in range(self.s):
                if self.control[i] == 1:
                    policy_0[i, :] = policy[i, :]
                else:
                    policy_1[i, :] = policy[i, :]
            res_policy_list1 = res_policy_list1 + [policy_1]
            res_policy_list0 = res_policy_list0 + [policy_0]
            ## calcul de la qfunction suivante
            v0 = np.asarray([np.vdot(Q_function_0[j, :], policy_0[j, :]) for j in range(self.s)])
            v1 = np.asarray([np.vdot(Q_function_1[j, :], policy_1[j, :]) for j in range(self.s)])
            for i, j in np.ndindex((self.s, self.a)):
                Q_function_0[i, j] = self.reward0[i, j] + gamma * np.vdot(v0, self.kernel[:, i, j])
                Q_function_1[i, j] = self.reward1[i, j] + gamma * np.vdot(v1, self.kernel[:, i, j])
        return res_policy_list0, res_policy_list1

    def policy_best_response(self, policy_list, gamma):
        policy_list__0 = [np.copy(policy) for policy in policy_list]
        Q_non_stat__0, Q_non_stat1 = self.policy_evaluation_exact_Q(policy_list__0, gamma) # evaluation of the policy of policy_list__0
        policy_list_0, policy_useless = self.greedy_best_response(policy_list__0, Q_non_stat__0, Q_non_stat1, gamma)
        Q_non_stat_0, Q_non_stat1 = self.policy_evaluation_exact_Q(policy_list_0, gamma) # evaluation of the policy of policy_list_0

        policy_list__1 = [np.copy(policy) for policy in policy_list]
        Q_non_stat0, Q_non_stat__1 = self.policy_evaluation_exact_Q(policy_list__1, gamma)# evaluation of the policy of policy_list__1
        policy_useless, policy_list_1 = self.greedy_best_response(policy_list__1, Q_non_stat0, Q_non_stat__1, gamma)
        Q_non_stat0, Q_non_stat_1 = self.policy_evaluation_exact_Q(policy_list_1, gamma)# evaluation of the policy of policy_list_1
        # print "max et min"
        # print np.max(Q_non_stat__0 - Q_non_stat_0)
        # print np.min(Q_non_stat__0 - Q_non_stat_0)
        # print np.max(Q_non_stat__1 - Q_non_stat_1)
        # print np.min(Q_non_stat__1 - Q_non_stat_1)

        cond = False
        i=0
        # print "On fait policy iteration"
        while (not cond) and not((np.linalg.norm(Q_non_stat_0 - Q_non_stat__0) < pow(10, -12)) and (np.linalg.norm(Q_non_stat_1 - Q_non_stat__1) < pow(10, -12))):
            i=i+1
            # print i
            # if i>0:
                # print "diff politiques"
                # print np.linalg.norm(Q_non_stat__0 - Q_non_stat_0)
                # print np.linalg.norm(Q_non_stat__1 - Q_non_stat_1)
                # print not((np.linalg.norm(Q_non_stat_0 - Q_non_stat__0) < pow(10, -12)) and (np.linalg.norm(Q_non_stat_1 - Q_non_stat__1) < pow(10, -12)))

            policy_list__0 = [np.copy(policy) for policy in policy_list_0]
            Q_non_stat__0, Q_non_stat1 = self.policy_evaluation_exact_Q(policy_list__0, gamma) # evaluation of the policy of policy_list__0
            policy_list_0, policy_useless = self.greedy_best_response(policy_list__0, Q_non_stat__0, Q_non_stat1, gamma)
            Q_non_stat_0, Q_non_stat1 = self.policy_evaluation_exact_Q(policy_list_0, gamma) # evaluation of the policy of policy_list_0
            list_cond0 = [np.array_equal(policy, policy_) for policy, policy_ in zip(policy_list__0, policy_list_0)]


            policy_list__1 = [np.copy(policy) for policy in policy_list_1]
            Q_non_stat0, Q_non_stat__1 = self.policy_evaluation_exact_Q(policy_list__1, gamma)# evaluation of the policy of policy_list__1
            policy_useless, policy_list_1 = self.greedy_best_response(policy_list__1, Q_non_stat0, Q_non_stat__1, gamma)
            Q_non_stat0, Q_non_stat_1 = self.policy_evaluation_exact_Q(policy_list_1, gamma)# evaluation of the policy of policy_list_1
            list_cond1 = [np.array_equal(policy, policy_) for policy, policy_ in zip(policy_list__1, policy_list_1)]

            cond = True
            for b,c in zip(list_cond0,list_cond1):
                cond = cond and b and c
            # if i>0:
            #     print "max et min"
            #     print np.max(Q_non_stat__0 - Q_non_stat_0)
            #     print np.min(Q_non_stat__0 - Q_non_stat_0)
            #     print np.max(Q_non_stat__1 - Q_non_stat_1)
            #     print np.min(Q_non_stat__1 - Q_non_stat_1)
            #     print not((np.linalg.norm(Q_non_stat_0 - Q_non_stat__0) < pow(10, -12)) and (np.linalg.norm(Q_non_stat_1 - Q_non_stat__1) < pow(10, -12)))
            # print not cond
            # print "--------------------------------------------"

        return policy_list__0, policy_list__1

    def exact_best_response_v(self, policy_list, gamma):
        policy_list__0, policy_list__1 = self.policy_best_response(policy_list, gamma)
        v0, v_useless = self.policy_evaluation_exact_v(policy_list__0, gamma)
        v_useless, v1 = self.policy_evaluation_exact_v(policy_list__1, gamma)
        return v0, v1

    def exact_best_response_Q(self, policy_list, gamma):
        policy_list__0, policy_list__1 = self.policy_best_response(policy_list, gamma)
        Q0, Q_useless = self.policy_evaluation_exact_Q(policy_list__0, gamma)
        Q_useless, Q1 = self.policy_evaluation_exact_Q(policy_list__1, gamma)
        return Q0, Q1

    def Apply_bellman(self, policy_list, Q_function0, Q_function1, gamma):
        #print len(policy_list)
        Q_function_0 = np.copy(Q_function0)
        Q_function_1 = np.copy(Q_function1)
        for policy in policy_list:
            ## calcul de la qfunction suivante
            v0=np.asarray([np.vdot(Q_function_0[j, :], policy[j, :]) for j in range(self.s)])
            v1=np.asarray([np.vdot(Q_function_1[j, :], policy[j, :]) for j in range(self.s)])
            for i, j in np.ndindex((self.s, self.a)):
                Q_function_0[i, j] = self.reward0[i, j] + gamma * np.vdot(v0, self.kernel[:, i, j])
                Q_function_1[i, j] = self.reward1[i, j] + gamma * np.vdot(v1, self.kernel[:, i, j])
        return Q_function_0, Q_function_1


# ################################################### Generate Features
    def Q_func2vect(self,Q):
        vect_ = np.zeros((self.s*self.a,1))
        for i,j in np.ndindex((self.s, self.a)):
            vect_[i*(self.a)+j,0] = Q[i,j]
        return vect_

    def vect2Q_func(self,vect_):
        Q = np.zeros((self.s,self.a))
        for i,j in np.ndindex((self.s, self.a)):
            Q[i,j] = vect_[i*(self.a)+j,0]
        return Q



    def random_Phi(self,d):
        Phi = np.random.randn(self.s*self.a,d)
        Phi_orth, r = np.linalg.qr(Phi)
        return Phi_orth

########################################################

    def value_iteration(self, K, gamma):
        Q0 = np.zeros((self.s, self.a))
        Q1 = np.zeros((self.s, self.a))
        liste_policy = []
        for i in range(K):
            policy = self.greedy_policy(Q0, Q1)
            Q0, Q1 = self.Apply_bellman([policy], Q0, Q1, gamma)
            liste_policy = liste_policy + [policy]
        return liste_policy


    def value_iteration_app(self, K, sigma, gamma):
        Q0 = np.zeros((self.s, self.a))
        Q1 = np.zeros((self.s, self.a))
        liste_policy = []
        for i in range(K):
            policy = self.greedy_policy(Q0, Q1)
            Q0, Q1 = self.Apply_bellman([policy], Q0, Q1, gamma)
            Q0 = Q0 + sigma * np.random.randn(self.s, self.a)
            Q1 = Q1 + sigma * np.random.randn(self.s, self.a)
            liste_policy = liste_policy + [policy]
        return liste_policy

    def merge_policy(self, pi0, pi1):
        policy = np.zeros((self.s, self.a))
        for i in range(self.s):
            if self.control[i] == 1:
                policy[i, :] = pi1[i, :]
            else:
                policy[i, :] = pi0[i, :]
        return policy

    def l2(self, estimate, target):
        return np.linalg.norm(estimate - target)/np.linalg.norm(target)


    def l2errorDiffQstarQpi(self, policy, gamma):
        Qstar0, Qstar1   = self.exact_best_response_Q([policy], gamma)
        Qpi0, Qpi1 = self.policy_evaluation_exact_Q([policy], gamma)
        return self.l2(Qpi0, Qstar0), self.l2(Qpi1, Qstar1)



####################################################################################################


def garnet_gen_sa( Ns, Na, Nb, sparsity, neighbor):
    ### generating the Kernel
    kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
    for i, j in np.ndindex((Ns, Na)):
        echantillon = rd.sample(list(set(range(Ns)).intersection(range(i-neighbor,i+neighbor))), Nb)
        cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
        for k in range(Nb):
            kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]
    ### generating rewards at random
    reward0 = np.random.randn(Ns, Na)
    reward1 = np.random.randn(Ns, Na)

    masque_reward = np.zeros((Ns, Na))
    N_sparsity = int(Ns * sparsity)
    i = 0
    while i < N_sparsity:
        i_ = rd.randint(0, Ns - 1)
        if masque_reward[i_, 0] == 0:
            masque_reward[i_, :] = 1
            i += 1
    reward0 = reward0 * masque_reward
    reward1 = reward1 * masque_reward
    control = np.random.randint(2, size=Ns)

    return Ns, Na, kernel, reward0, reward1, control



def garnet_gen_s( Ns, Na, Nb, sparsity, neighbor):
    ### generating the Kernel
    kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
    for i, j in np.ndindex((Ns, Na)):
        echantillon = rd.sample(list(set(range(Ns)).intersection(range(i-neighbor,i+neighbor))), Nb)
        cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
        for k in range(Nb):
            kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]
    ### generating rewards at random
    reward0 = np.zeros((Ns, Na))
    reward1 = np.zeros((Ns, Na))

    biais0 = np.random.randn(Ns)
    biais1 = np.random.randn(Ns)
    for i, j in np.ndindex((Ns, Na)):
        reward0[i,j] = biais0[i]
        reward1[i,j] = biais1[i]

    masque_reward = np.zeros((Ns, Na))
    N_sparsity = int(Ns * sparsity)
    i = 0
    while i < N_sparsity:
        i_ = rd.randint(0, Ns - 1)
        if masque_reward[i_, 0] == 0:
            masque_reward[i_, :] = 1
            i += 1
    reward0 = reward0 * masque_reward
    reward1 = reward1 * masque_reward
    control = np.random.randint(2, size=Ns)

    return Ns, Na, kernel, reward0, reward1, control

def garnet_gen_s_linear( Ns, Na, Nb, sparsity, neighbor):
    ### generating the Kernel
    kernel = np.zeros((Ns, Ns, Na))  # p(s'|s,a)
    for i, j in np.ndindex((Ns, Na)):
        echantillon = rd.sample(list(set(range(Ns)).intersection(range(i-neighbor,i+neighbor))), Nb)
        cumulative = np.concatenate(([0], sort([rd.random() for k in range(Nb - 1)]), [1]), axis=0)
        for k in range(Nb):
            kernel[echantillon[k], i, j] = cumulative[k + 1] - cumulative[k]
    ### generating rewards at random
    reward0 = np.zeros((Ns, Na))
    reward1 = np.zeros((Ns, Na))

    biais0 = (np.arange(Ns)/(1.0*(Ns-1)))
    biais1 = 1-(np.arange(Ns)/(1.0*(Ns-1)))

    for i, j in np.ndindex((Ns, Na)):
        reward0[i,j] = biais0[i]
        reward1[i,j] = biais1[i]

    masque_reward = np.zeros((Ns, Na))
    N_sparsity = int(Ns * sparsity)
    i = 0
    while i < N_sparsity:
        i_ = rd.randint(0, Ns - 1)
        if masque_reward[i_, 0] == 0:
            masque_reward[i_, :] = 1
            i += 1
    reward0 = reward0 * masque_reward
    reward1 = reward1 * masque_reward
    control = np.random.randint(2, size=Ns)

    return Ns, Na, kernel, reward0, reward1, control









####################################################################################################


    # def l2errorBellmanResidual(self, Q, gamma):
    #     policy = self.greedy_policy(Q)
    #     TQ = self.Apply_bellman([policy],Q,gamma)
    #     return self.l2(TQ, Q)



########################################################
# def random_distr(l):
#     r = random.uniform(0, 1)
#     s = 0
#     for item, prob in l:
#         s += prob
#         if s >= r:
#             return item
#     return l[-1]
#
#
# def sample_action_unif(action_set_list):
#     action_unif_list = []
#     for a_List in action_set_list:
#         action_unif_list.append(rd.sample(a_List, 1)[0])
#     return action_unif_list
#
#
# def sample_action(action_set_list, policy_list):
#     action_list = []
#     for a_List, pi_list in zip(action_set_list, policy_list):
#         action_list.append(random_distr(zip(a_List, pi_list)))
#     return action_list
#
#
# def split(X, n):
#     X_ = []
#     while len(X) > n:
#         X_.append(X[:n])
#         X = X[n:]
#     X_.append(X)
#     return X_
#
#
# def merge(X):
#     X_ = []
#     while len(X) > 0:
#         X_ = X_ + X[0]
#         X = X[1:]
#     return X_
#
#
# def merge_(X):
#     X_ = []
#     len_x = []
#     for x in X:
#         X_ = X_ + x
#         len_x.append(len(x))
#     return X_, len_x
#
#
# def split_(X, len_x):
#     X_ = []
#     while len(len_x) > 0:
#         X_.append(X[:len_x[0]])
#         X = X[len_x[0]:]
#         len_x = len_x[1:]
#     return X_