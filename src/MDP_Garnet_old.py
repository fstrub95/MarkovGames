import sys
import numpy as np
import random as rd
from scipy import *
from math import log
import function_approximation_extratree_ as fa
#import function_approximation_tabular_ as fa
from itertools import product
BETA1 = 0.01
BETA2 = 0.99

rd.seed()


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

    def action_set_list(self, s):
        return [range(self.a)]

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
            state_action = [next_s, sample_action(self.action_set_list(next_s),
                                                  [list((policy_vect[(-i-1) % len(policy_vect)])[next_s, :])])]
            #print (i % len(policy_vect))
            #print (policy_vect[i % len(policy_vect)])[next_s, :]
            #print state_action
        # print rollout[0]
        return rollout

    def uniform_batch_data(self,N):
        ech = []
        print i
        for j in range(N):
            state = self.sample_state_unif()
            action_list = self.action_set_list(state)[0]
            action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
            # print action
            for k in range(n_resample):
                ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
        # sars_a_ = [([s,a],r,[s_,a_]) for [[s,[a],r],[s_,[a_],r_]] in ech]
        sars_ = [(s, a, r, s_) for [[s, [a], r], [s_, [a_], r_]] in ech]
        return sars_

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



################################################### Generate Features
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


####################################################

###################################################

    def LSPI_MDP(self, Phi, n_iteration, gamma, alpha):
        omega = np.zeros((size(Phi,1),1))
        for k in range(n_iteration):
            #print np.dot(Phi, omega)
            Q = self.vect2Q_func(np.dot(Phi, omega))
            policy_ = self.greedy_policy(Q)
            reward_ = self.Q_func2vect(self.reward)
            ker_ = np.zeros((self.s*self.a,self.s*self.a))
            for i,j,i_,j_ in np.ndindex((self.s, self.a, self.s, self.a)):
                ker_[i*(self.a)+j, i_*(self.a)+j_] = self.kernel[i_,i,j]*policy_[i_,j_]
            Id = np.zeros((self.s*self.a, self.s*self.a))
            for i in range(self.s*self.a):
                Id[i,i]=1
            A = np.dot(np.transpose(Phi), np.dot(Id-gamma*ker_, Phi))
            #print A
            b= np.dot(np.transpose(Phi), reward_)
            omega =(1-alpha)*omega +alpha*np.linalg.solve(A,b)
        return self.vect2Q_func(np.dot(Phi, omega)), policy_

    def PI_with_BR(self, Phi, n_iteration, gamma, alpha):
        omega = np.zeros((size(Phi,1),1))
        for k in range(n_iteration):
            #print np.dot(Phi, omega)
            Q = self.vect2Q_func(np.dot(Phi, omega))
            policy_ = self.greedy_policy(Q)
            reward_ = self.Q_func2vect(self.reward)
            ker_ = np.zeros((self.s*self.a,self.s*self.a))
            for i,j,i_,j_ in np.ndindex((self.s, self.a, self.s, self.a)):
                ker_[i*(self.a)+j, i_*(self.a)+j_] = self.kernel[i_,i,j]*policy_[i_,j_]
            Id = np.zeros((self.s*self.a, self.s*self.a))
            for i in range(self.s*self.a):
                Id[i,i]=1
            A = np.dot(Id-gamma*ker_, Phi)
            #print A
            b = np.dot(np.transpose(A), reward_)
            A = np.dot(np.transpose(A),A)
            omega = (1-alpha)*omega +alpha*np.linalg.solve(A,b)
        return self.vect2Q_func(np.dot(Phi, omega)), policy_

    def P_R_omega(self, omega, Phi, slist, rlist, s_list, gamma):
        Q = self.vect2Q_func(np.dot(Phi, omega))
        policy_ = self.greedy_policy(Q)
        reward_ = np.zeros((self.s*self.a,1))
        ker_ = np.zeros((self.s*self.a,self.s*self.a))
        for [s, a1],[s_, a1_] in zip(slist, s_list):
            for j_ in range(self.a):
                ker_[s*self.a+a1, s_*self.a+j_] -= gamma*policy_[s_,j_]
            ker_[s*(self.a)+a1, s*(self.a)+a1] += 1
        for [s, a1], r in zip(slist, rlist):
            reward_[s*(self.a)+a1] += r
        return ker_, reward_

    def residual(self, omega, Phi, slist, rlist, s_list, gamma):
        ker_, reward_ = self.P_R_omega(omega, Phi, slist, rlist, s_list, gamma)
        return np.dot(ker_,np.dot(Phi, omega)) - reward_

    def projected_residual(self, omega, Phi, slist, rlist, s_list, gamma):
        resi = self.residual(omega, Phi, slist, rlist, s_list, gamma)
        return np.linalg.solve(np.dot(np.transpose(Phi), Phi), np.dot(np.transpose(Phi), resi))

    def gradient_POBR(self, omega, Phi, slist, rlist, s_list, gamma):
        PP= np.dot(np.transpose(Phi),Phi)
        PP=np.dot(PP,PP)
        ker_, reward_ = self.P_R_omega(omega, Phi, slist, rlist, s_list, gamma)
        br = np.dot(np.transpose(Phi),np.dot(ker_,np.dot(Phi, omega)) - reward_)
        mtrx = np.dot(np.transpose(Phi), np.dot(np.transpose(ker_), Phi))
        return np.dot(mtrx,np.linalg.solve(PP,br))

    def gradient_OBR(self, omega, Phi, slist, rlist, s_list, gamma):
        ker_, reward_ = self.P_R_omega(omega, Phi, slist, rlist, s_list, gamma)
        br = np.dot(ker_,np.dot(Phi, omega)) - reward_
        mtrx = np.dot(np.transpose(Phi), np.transpose(ker_))
        return np.dot(mtrx,br)

    def LSPI_MDP_Batch(self, Q_star, Phi, n_iteration, gamma, Nsamples, alpha = -1, tau = 0.9, beta1 = BETA1, beta2 = BETA2, n_resample = 1):
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        list_res = []
        # nombre d'echantillons par iteration
        eta = 1.0

        ech = []
        for j in range(Nsamples):
            state = self.sample_state_unif()
            action_list = self.action_set_list(state)[0]
            action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
            # print action
            for k in range(n_resample):
                ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
        # sars_a_ = [([s,a],r,[s_,a_]) for [[s,[a],r],[s_,[a_],r_]] in ech]
        sars_a_ = [([s, a], r, [s_, a_]) for [[s, [a], r], [s_, [a_], r_]] in ech]
        # on retourne la liste
        slist, rlist, s_list = zip(*sars_a_)

        omega = np.zeros((size(Phi,1),1))
        for k in range(n_iteration):
            # print k
            #print np.dot(Phi, omega)
            Q_ = self.vect2Q_func(np.dot(Phi, omega))
            policy_ = self.greedy_policy(Q_)
            ker_, reward_ = self.P_R_omega(omega, Phi, slist, rlist, s_list, gamma)
            A = np.dot(np.transpose(Phi), np.dot(ker_, Phi))
            b= np.dot(np.transpose(Phi), reward_)
            p = self.gradient_POBR(omega, Phi, slist, rlist, s_list, gamma)
            g = np.linalg.solve(A,b)-omega
            if alpha == -1:
                norm_f_step = np.linalg.norm(self.projected_residual(omega+eta*g, Phi, slist, rlist, s_list, gamma))
                norm_f = np.linalg.norm(self.projected_residual(omega, Phi, slist, rlist, s_list, gamma))
                scalar_prod = np.dot(np.transpose(p),g)
                wolf_update = 0
                while (1 >= eta > 1e-10) and (not(norm_f_step < norm_f + eta*beta1*scalar_prod) or not(norm_f_step > norm_f + eta*beta2*scalar_prod)) and (wolf_update < -10/log(tau)):
                    # print "-------------------------"
                    # print "norm plus grad POBR"
                    # print np.linalg.norm(self.projected_residual(omega+eta*p, Phi, slist, rlist, s_list, gamma))
                    # print "norm POBR"
                    # print np.linalg.norm(self.projected_residual(omega, Phi, slist, rlist, s_list, gamma))
                    # print "norm POBR plus machin"
                    # print np.linalg.norm(self.projected_residual(omega, Phi, slist, rlist, s_list, gamma)) + eta*beta*np.dot(np.transpose(p), g)
                    # print "cos"
                    # print np.dot(np.transpose(p), g)
                    if not(norm_f_step < norm_f + eta*beta1*scalar_prod):
                        eta = eta*tau
                    if not(norm_f_step > norm_f + eta*beta2*scalar_prod) and (eta<1):
                        eta = eta/tau
                    print eta
                    norm_f_step = np.linalg.norm(self.projected_residual(omega+eta*g, Phi, slist, rlist, s_list, gamma))
                    wolf_update += 1
                #     print "iteration"
                #     print k
                #     print eta
                # print "norm POBR"
                # print np.linalg.norm(self.projected_residual(omega, Phi, slist, rlist, s_list, gamma))
                # print "eta"
                # print eta
                omega =omega +eta*g
            else:
                omega =omega +alpha*g
            list_res.append([policy_, omega, eta, np.linalg.norm(Q_star - self.policy_evaluation_exact_Q([policy_],gamma))])
        return self.vect2Q_func(np.dot(Phi, omega)), policy_, list_res

    def PI_with_BR_Batch(self, Q_star, Phi, n_iteration, gamma, Nsamples, alpha = -1, tau = 0.9, beta1 = BETA1, beta2 = BETA2, n_resample = 1):
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        # nombre d'echantillons par iteration
        list_res = []

        eta = 1.0

        ech = []
        for j in range(Nsamples):
            state = self.sample_state_unif()
            action_list = self.action_set_list(state)[0]
            action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
            # print action
            for k in range(n_resample):
                ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
        # sars_a_ = [([s,a],r,[s_,a_]) for [[s,[a],r],[s_,[a_],r_]] in ech]
        sars_a_ = [([s, a], r, [s_, a_]) for [[s, [a], r], [s_, [a_], r_]] in ech]
        # on retourne la liste
        slist, rlist, s_list = zip(*sars_a_)

        omega = np.zeros((size(Phi,1),1))
        for k in range(n_iteration):
            # print k
            #print np.dot(Phi, omega)
            Q_ = self.vect2Q_func(np.dot(Phi, omega))
            policy_ = self.greedy_policy(Q_)
            ker_, reward_ = self.P_R_omega(omega, Phi, slist, rlist, s_list, gamma)
            A = np.dot(ker_, Phi)
            #print A
            b = np.dot(np.transpose(A), reward_)
            A = np.dot(np.transpose(A), A)
            p = self.gradient_OBR(omega, Phi, slist, rlist, s_list, gamma)
            g = np.linalg.solve(A, b) - omega
            if alpha == -1:
                norm_f_step = np.linalg.norm(self.projected_residual(omega+eta*g, Phi, slist, rlist, s_list, gamma))
                norm_f = np.linalg.norm(self.projected_residual(omega, Phi, slist, rlist, s_list, gamma))
                scalar_prod = np.dot(np.transpose(p),g)
                wolf_update = 0
                while (1 >= eta > 1e-10) and (not(norm_f_step < norm_f + eta*beta1*scalar_prod) or not(norm_f_step > norm_f + eta*beta2*scalar_prod)) and (wolf_update < -10/log(tau)):
                    # print "-------------------------"
                    # print "norm plus grad OBR"
                    # print np.linalg.norm(self.residual(omega + eta * p, Phi, slist, rlist, s_list, gamma))
                    # print "norm OBR"
                    # print np.linalg.norm(self.residual(omega, Phi, slist, rlist, s_list, gamma))
                    # print "norm OBR plus machin"
                    # print np.linalg.norm(self.residual(omega, Phi, slist, rlist, s_list, gamma)) + eta*beta*np.dot(np.transpose(p), g)
                    # print "cos"
                    # print np.dot(np.transpose(p), g)
                    if not(norm_f_step < norm_f + eta*beta1*scalar_prod):
                        eta = eta*tau
                    if not(norm_f_step > norm_f + eta*beta2*scalar_prod):
                        eta = eta/tau
                    print eta
                    norm_f_step = np.linalg.norm(self.projected_residual(omega+eta*g, Phi, slist, rlist, s_list, gamma))
                    wolf_update += 1
                #     print "iteration"
                #     print k
                #     print eta
                # print "norm OBR"
                # print np.linalg.norm(self.residual(omega, Phi, slist, rlist, s_list, gamma))
                # print "eta"
                # print eta
                omega = omega + eta * g
            else:
                omega = omega +alpha*g

            list_res.append([omega, eta, np.linalg.norm(Q_star - self.policy_evaluation_exact_Q([policy_],gamma))])
        return self.vect2Q_func(np.dot(Phi, omega)), policy_, list_res






####################################################









    def value_iteration(self, gamma, n_echantillon_total, n_iteration, n_resample = 1):
        # policy est la politique pour echantilloner les rollout
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        # policy est la politique de la valeur courante
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        # nombre d'echantillons par iteration
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
            # sars_a_ = [([s,a],r,[s_,a_]) for [[s,[a],r],[s_,[a_],r_]] in ech]
            sars_a_ = [([s, a], r, [s_, a_]) for [[s, [a], r], [s_, [a_], r_]] in ech]
            # on retourne la liste
            slist, rlist, s_list = zip(*sars_a_)
            value_list = reg.predict(list(s_list))
            X = list(slist)
            # print list(s_list)
            Y = [r + gamma * v for r, v in zip(list(rlist), value_list)]
            reg.regression([x[0] for x in split(X,n_resample)], [mean(y) for y in split(Y,n_resample)])
            q = self.regressor_to_Q_function(reg)
            policy__ = self.greedy_policy(q)
        return [policy__]

    def value_iteration_non_stationary(self, gamma, n_echantillon_total, n_iteration, n_resample = 1):
        # policy est la politique pour echantilloner les rollout
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        # policy est la politique de la valeur courante
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        list_policy = [policy_]
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        # nombre d'echantillons par iteration
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
            # sars_a_ = [([s,a],r,[s_,a_]) for [[s,[a],r],[s_,[a_],r_]] in ech]
            sars_a_ = [([s, a], r, [s_, a_]) for [[s, [a], r], [s_, [a_], r_]] in ech]
            # on retourne la liste
            slist, rlist, s_list = zip(*sars_a_)
            value_list = reg.predict(list(s_list))
            X = list(slist)
            # print list(s_list)
            Y = [r + gamma * v for r, v in zip(list(rlist), value_list)]
            reg.regression([x[0] for x in split(X,n_resample)], [mean(y) for y in split(Y,n_resample)])
            q = self.regressor_to_Q_function(reg)
            policy__ = self.greedy_policy(q)
            list_policy = list_policy + [policy__]
        return list_policy

    def value_iteration_non_stationary_err(self, gamma, n_echantillon_total, n_iteration, n_resample = 1):
        # policy est la politique pour echantilloner les rollout
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        # policy est la politique de la valeur courante
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        list_policy = [policy_]
        list_err = []
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        # nombre d'echantillons par iteration
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
            # sars_a_ = [([s,a],r,[s_,a_]) for [[s,[a],r],[s_,[a_],r_]] in ech]
            sars_a_ = [([s, a], r, [s_, a_]) for [[s, [a], r], [s_, [a_], r_]] in ech]
            # on retourne la liste
            slist, rlist, s_list = zip(*sars_a_)
            value_list = reg.predict(list(s_list))
            X = list(slist)
            # print list(s_list)
            Y = [r + gamma * v for r, v in zip(list(rlist), value_list)]
            q_ = self.Apply_bellman([policy__],self.regressor_to_Q_function(reg),gamma)
            reg.regression([x[0] for x in split(X,n_resample)], [mean(y) for y in split(Y,n_resample)])
            q = self.regressor_to_Q_function(reg)
            policy__ = self.greedy_policy(q)
            list_policy = list_policy + [policy__]
            list_err = list_err + [q_ - q]
        return list_policy, list_err

    def value_iteration_non_stationary_err_fair(self, gamma, n_echantillon_total, n_iteration, n_resample = 1):
        # policy est la politique pour echantilloner les rollout
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        # policy est la politique de la valeur courante
        policy__ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        list_policy = [policy_]
        list_err = []
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        # nombre d'echantillons par iteration
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n*(i+1)):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], 2, [policy__]))  # uniform sampling
            # sars_a_ = [([s,a],r,[s_,a_]) for [[s,[a],r],[s_,[a_],r_]] in ech]
            sars_a_ = [([s, a], r, [s_, a_]) for [[s, [a], r], [s_, [a_], r_]] in ech]
            # on retourne la liste
            slist, rlist, s_list = zip(*sars_a_)
            value_list = reg.predict(list(s_list))
            X = list(slist)
            # print list(s_list)
            Y = [r + gamma * v for r, v in zip(list(rlist), value_list)]
            q_ = self.Apply_bellman([policy__],self.regressor_to_Q_function(reg),gamma)
            reg.regression([x[0] for x in split(X,n_resample)], [mean(y) for y in split(Y,n_resample)])
            q = self.regressor_to_Q_function(reg)
            policy__ = self.greedy_policy(q)
            list_policy = list_policy + [policy__]
            list_err = list_err + [q_ - q]
        return list_policy, list_err

    def psdp(self, gamma, n_echantillon_total, n_iteration, n_resample = 1):
        # policy est la politique pour echantilloner les rollout
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        policy_list = [policy_]
        # n = int((2.0 * n_echantillon_total) / (1.0 * n_iteration*(n_iteration-1)))
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], i + 2, policy_list))  # uniform sampling
            discounted_sum_reward = []
            for e in ech:
                reward = 0
                #print "length"
                #print len(e)
                #print len(policy_list)
                for [s, [a], r] in e[(i)::-1]:
                    #print "reward er r"
                    #print reward
                    #print r
                    reward = r + gamma * reward
                discounted_sum_reward.append(reward)
            slist = [[e[0][0], e[0][1][0]] for e in ech]
            #print slist
            # print len(ech[0])
            reg.regression([x[0] for x in split(slist,n_resample)], [mean(y) for y in split(discounted_sum_reward,n_resample)])
            q = self.regressor_to_Q_function(reg)
            policy__ = self.greedy_policy(q)
            policy_list = policy_list + [policy__]#ajouter la politique en debut de liste
        return policy_list

    def psdp_err(self, gamma, n_echantillon_total, n_iteration, n_resample = 1):
        # policy est la politique pour echantilloner les rollout
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        policy_list = [policy_]
        # n = int((2.0 * n_echantillon_total) / (1.0 * n_iteration*(n_iteration-1)))
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        list_err = []
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], i + 2, policy_list))  # uniform sampling
            discounted_sum_reward = []
            for e in ech:
                reward = 0
                #print "length"
                #print len(e)
                #print len(policy_list)
                for [s, [a], r] in e[(i)::-1]:
                    #print "reward er r"
                    #print reward
                    #print r
                    reward = r + gamma * reward
                discounted_sum_reward.append(reward)
            slist = [[e[0][0], e[0][1][0]] for e in ech]
            #print slist
            # print len(ech[0])
            reg.regression([x[0] for x in split(slist,n_resample)], [mean(y) for y in split(discounted_sum_reward,n_resample)])
            q = self.regressor_to_Q_function(reg)
            q_ = self.Apply_bellman(policy_list, np.zeros((self.s,self.a)),gamma)
            list_err = list_err + [q-q_]
            policy__ = self.greedy_policy(q)
            policy_list = policy_list + [policy__]#ajouter la politique en debut de liste
        return policy_list, list_err

    def policy_iteration_non_stationary(self, gamma, n_echantillon_total, n_iteration, epsilon, n_resample = 1):
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        policy_list = [policy_]
        # n = int((2.0 * n_echantillon_total) / (1.0 * n_iteration*(n_iteration-1)))
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], i + 2, policy_list))  # uniform sampling
            ## building reward list
            discounted_sum_reward = []
            for e in ech:
                reward = 0
                #print "length"
                #print len(e)
                #print len(policy_list)
                for [s, [a], r] in e[(i)::-1]:
                    #print "reward er r"
                    #print reward
                    #print r
                    reward = r + gamma * reward
                discounted_sum_reward.append(reward)
            ## state action list
            slist = [[e[0][0], e[0][1][0]] for e in ech]
            ## next state liste
            s_list = [[[e[-1][0],a] for a in self.action_set_list(e[-1][0])[0]] for e in ech]
            s_list, len_s_list = merge_(s_list)
            #print (int(log(epsilon)/(log(gamma)*len(policy_list)))+3)
            for k in range(int(log(epsilon)/(log(gamma)*len(policy_list))) + 3):
                value_list = reg.predict(s_list)
                value_list = split_(value_list, len_s_list)
                next_value = [r + (gamma**(i+1)) * np.vdot(np.asarray(v), (policy_list[0])[s_[0],:]) for s_,r,v in zip(s_list,discounted_sum_reward,value_list)]
                reg.regression([x[0] for x in split(slist,n_resample)], [mean(y) for y in split(next_value,n_resample)])
            ## politique greedy
            q = self.regressor_to_Q_function(reg)
            policy__ = self.greedy_policy(q)
            policy_list = policy_list + [np.copy(policy__)] #ajouter la politique en debut de liste
        return policy_list

    def policy_iteration_non_stationary_fixed_length(self, gamma, n_echantillon_total, n_iteration, len_non_stationarry, epsilon, n_resample = 1):
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        policy_list = [policy_]
        # n = int((2.0 * n_echantillon_total) / (1.0 * n_iteration*(n_iteration-1)))
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], i + 2, policy_list))  # uniform sampling
            ## building reward list
            discounted_sum_reward = []
            for e in ech:
                reward = 0
                #print "length"
                #print len(e)
                #print len(policy_list)
                for [s, [a], r] in e[(i)::-1]:
                    #print "reward er r"
                    #print reward
                    #print r
                    reward = r + gamma * reward
                discounted_sum_reward.append(reward)
            ## state action list
            slist = [[e[0][0], e[0][1][0]] for e in ech]
            ## next state liste
            s_list = [[[e[-1][0],a] for a in self.action_set_list(e[-1][0])[0]] for e in ech]
            s_list, len_s_list = merge_(s_list)
            #Evaluation of the non stationarry policy
            l=min(len_non_stationarry,i+1)
            policy_list_fixed_size = policy_list[-l:]
            #print (int(log(epsilon)/(log(gamma)*len(policy_list_fixed_size)))+3)
            for k in range(int(log(epsilon)/(log(gamma)*len(policy_list_fixed_size))) + 3):
                value_list = reg.predict(s_list)
                value_list = split_(value_list, len_s_list)
                next_value = [r + (gamma**(i+1)) * np.vdot(np.asarray(v), (policy_list_fixed_size[0])[s_[0],:]) for s_,r,v in zip(s_list,discounted_sum_reward,value_list)]
                reg.regression([x[0] for x in split(slist,n_resample)], [mean(y) for y in split(next_value,n_resample)])
            ## politique greedy
            q = self.regressor_to_Q_function(reg)
            policy__ = self.greedy_policy(q)
            policy_list = policy_list + [np.copy(policy__)] #ajouter la politique en debut de liste
        return policy_list

    def policy_iteration_non_stationary_fixed_length_err(self, gamma, n_echantillon_total, n_iteration, len_non_stationarry, epsilon, n_resample = 1):
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        policy_list = [policy_]
        # n = int((2.0 * n_echantillon_total) / (1.0 * n_iteration*(n_iteration-1)))
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        list_err = []
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], 2, policy_list))  # uniform sampling
            ## building reward list
            discounted_sum_reward = []
            for e in ech:
                reward = 0
                #print "length"
                #print len(e)
                #print len(policy_list)
                discounted_sum_reward.append(e[0][2])
            ## state action list
            slist = [[e[0][0], e[0][1][0]] for e in ech]
            # print discounted_sum_reward
            ## next state liste
            s_list = [e[-1][0] for e in ech]
            #Evaluation of the non stationarry policy
            l=min(len_non_stationarry,i+1)
            policy_list_fixed_size = policy_list[-l:]
            print len(policy_list_fixed_size)
            print ((l)*(int(log(epsilon)/(log(gamma)))/(l))+l)
            current_policy = policy_list_fixed_size[(-k-1) % len(policy_list_fixed_size)]
            for k in range(((l)*(int(log(epsilon)/(log(gamma)))/(l))+l)):
                current_policy = policy_list_fixed_size[(k) % len(policy_list_fixed_size)]
                s_a_list = [[s_,np.argmax(current_policy[s_,:])] for s_ in s_list]
                value_list = reg.predict(s_a_list)
                next_value = [r + gamma * v for r,v in zip(discounted_sum_reward,value_list)]
                reg.regression(slist, next_value)
            ## politique greedy
            q = self.regressor_to_Q_function(reg)
            q_ = self.policy_evaluation_exact_Q(policy_list_fixed_size, gamma)
            list_err = list_err + [q - q_]
            policy__ = self.greedy_policy(q)
            policy_list = policy_list + [np.copy(policy__)] #ajouter la politique en debut de liste
        return policy_list, list_err

    def policy_iteration_non_stationary_fixed_length_err_fair(self, gamma, n_echantillon_total, n_iteration, len_non_stationarry, epsilon, n_resample = 1):
        policy_ = (1.0 * np.ones((self.s, self.a))) / (1.0 * self.a)
        policy_list = [policy_]
        # n = int((2.0 * n_echantillon_total) / (1.0 * n_iteration*(n_iteration-1)))
        n = int((1.0 * n_echantillon_total) / (1.0 * n_iteration))
        list_err = []
        # regresseur utilise pour approximer la q-fonction
        reg = fa.function_approx([self.s,self.a])
        for i in range(n_iteration):
            ech = []
            print i
            for j in range(n*(i+1)):
                state = self.sample_state_unif()
                action_list = self.action_set_list(state)[0]
                action = random_distr(zip(action_list, [1.0 / (1.0 * len(action_list)) for k in action_list]))
                # print action
                for k in range(n_resample):
                    ech.append(self.rollout(state, [action], 2, policy_list))  # uniform sampling
            ## building reward list
            discounted_sum_reward = []
            for e in ech:
                reward = 0
                #print "length"
                #print len(e)
                #print len(policy_list)
                discounted_sum_reward.append(e[0][2])
            ## state action list
            slist = [[e[0][0], e[0][1][0]] for e in ech]
            # print discounted_sum_reward
            ## next state liste
            s_list = [e[-1][0] for e in ech]
            #Evaluation of the non stationarry policy
            l=min(len_non_stationarry,i+1)
            policy_list_fixed_size = policy_list[-l:]
            print len(policy_list_fixed_size)
            print ((l)*(int(log(epsilon)/(log(gamma)))/(l))+l)
            current_policy = policy_list_fixed_size[(-k-1) % len(policy_list_fixed_size)]
            for k in range(((l)*(int(log(epsilon)/(log(gamma)))/(l))+l)):
                current_policy = policy_list_fixed_size[(k) % len(policy_list_fixed_size)]
                s_a_list = [[s_,np.argmax(current_policy[s_,:])] for s_ in s_list]
                value_list = reg.predict(s_a_list)
                next_value = [r + gamma * v for r,v in zip(discounted_sum_reward,value_list)]
                reg.regression(slist, next_value)
            ## politique greedy
            q = self.regressor_to_Q_function(reg)
            q_ = self.policy_evaluation_exact_Q(policy_list_fixed_size, gamma)
            list_err = list_err + [q - q_]
            policy__ = self.greedy_policy(q)
            policy_list = policy_list + [np.copy(policy__)] #ajouter la politique en debut de liste
        return policy_list, list_err

    def regressor_to_Q_function(self, reg):
        l = product(range(self.s), range(self.a))
        list_states = [list(sa) for sa in l]
        value_list = reg.predict(list_states)
        Q_function = np.ones((self.s, self.a))
        for sa, v in zip(list_states, value_list):
            Q_function[sa[0], sa[1]] = v
        return Q_function

    def greedy_policy(self,Q_function):
        policy = np.zeros(((Q_function[:, 0]).size, (Q_function[0, :]).size))
        for i in range((Q_function[:, 0]).size):
            policy[i, np.argmax(Q_function[i, :], axis=0)] = 1
        return policy


def random_distr(l):
    r = random.uniform(0, 1)
    s = 0
    for item, prob in l:
        s += prob
        if s >= r:
            return item
    return l[-1]


def sample_action_unif(action_set_list):
    action_unif_list = []
    for a_List in action_set_list:
        action_unif_list.append(rd.sample(a_List, 1)[0])
    return action_unif_list


def sample_action(action_set_list, policy_list):
    action_list = []
    for a_List, pi_list in zip(action_set_list, policy_list):
        action_list.append(random_distr(zip(a_List, pi_list)))
    return action_list


def split(X, n):
    X_ = []
    while len(X) > n:
        X_.append(X[:n])
        X = X[n:]
    X_.append(X)
    return X_


def merge(X):
    X_ = []
    while len(X) > 0:
        X_ = X_ + X[0]
        X = X[1:]
    return X_


def merge_(X):
    X_ = []
    len_x = []
    for x in X:
        X_ = X_ + x
        len_x.append(len(x))
    return X_, len_x


def split_(X, len_x):
    X_ = []
    while len(len_x) > 0:
        X_.append(X[:len_x[0]])
        X = X[len_x[0]:]
        len_x = len_x[1:]
    return X_

# gamma = 0.9

# gar = Garnet_MDP(20, 5, 4, 0.7)
### politique exact
# pi = gar.policy_iteration_exact(gamma)
# print pi[0]
# print gar.policy_evaluation_exact_Q(pi, gamma)
###politique approchee par value iteration
# policy = gar.value_iteration(gamma,500000,20)
# print gar.policy_evaluation_exact_Q(pi, gamma)
# print gar.policy_evaluation_exact_Q(policy, gamma)
# print pi[0]
###politique approchee par psdp
# policy = gar.psdp(gamma, 400000, 200)
# print gar.policy_evaluation_exact_Q(pi, gamma)
# print gar.policy_evaluation_exact_Q([policy[-1]], gamma)
# print gar.policy_evaluation_exact_Q(policy, gamma)
# print pi[0]
# print policy[-1]
###politique approchee par value iteration non-stationnaire
# policy = gar.value_iteration_non_stationary(gamma,50000,20,10)
# print len(policy)
# print gar.policy_evaluation_exact_Q(pi, gamma)
# print gar.policy_evaluation_exact_Q(policy, gamma)
# print pi[0]
###policy_iteration_non_stationarry
# policy = gar.policy_iteration_non_stationary(gamma,500000,1000,0.001)
# print len(policy)
# print gar.policy_evaluation_exact_Q(pi, gamma)
# print gar.policy_evaluation_exact_Q(policy, gamma)
# print pi[0]
