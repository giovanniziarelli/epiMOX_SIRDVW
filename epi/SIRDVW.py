#import pdb; pdb.set_trace()
import numpy as np
import pandas as pd
import datetime
import jax
import sys
import jax.numpy as jnp
from scipy.optimize import Bounds
import scipy.optimize as scopt
import matplotlib.pyplot as plt
from optimparallel import minimize_parallel
from epi import parameters_const as pm

# SIRDVW model class
class SIRDVW:

    def __init__(self, Nc, params, t_list, DPC_start, DPC_end, data, Pop, 
                       by_age, codes, vaccines, Delta_t, DO, out_path, out_type='h5'):#added vaccines with respect to SIR
        # initialize compartments
        # Y0: Nc x Ns
        self.Y0 = np.zeros((Nc, Pop.size)).squeeze()
        # Y: Nc x T x Ns
        self.Y = np.zeros((Nc, t_list[-1]+1, Pop.size)).squeeze()
        #initialize parameters
        self.params = params
        self.Ns = Pop.size
        self.Nc = Nc

        self.t_list = t_list
        self.DPC_start = DPC_start
        self.DPC_end = DPC_end

        self.data = data

        self.forecast = False
        
        self.Pop = Pop
        self.r_a = np.array([0.33, 1, 1, 1, 1.47, 1.47]) if by_age else 1 
        self.sigma1 = 0.21 
        self.sigma2 = 0.11 
        self.theta1 = 0.197 
        self.theta2 = 0.036
        self.f = np.array([1e-4, 6e-4, 4.5e-3, 2.3e-2, 4.3e-2, 1.7e-1]) if by_age else 0.0115 
        self.t_rec = 14.20  
        self.US1 = np.reshape(vaccines['prima_dose'].to_numpy(), (9, len(t_list) ), order = 'F') 
        self.US2 = np.reshape(vaccines['seconda_dose'].to_numpy() + vaccines['mono_dose'].to_numpy(), (9, len(t_list) ), order = 'F') 
        self.UR = np.reshape(vaccines['pregressa_infezione'].to_numpy(), (9, len(t_list) ), order = 'F') 
        self.period_U = 7
        self.Delta_t = np.clip(Delta_t * 8 , 0, 1) 
        self.Sfrac = np.zeros((t_list[-1]+1, Pop.size)).squeeze()

        self.by_age = by_age 
        self.codes = codes

        self.ages_opt_fixed =[]
        self.ages_opt = []
        self.Ns_opt = 0

        self.DO = DO /np.max(np.max(DO))
       
        # Flag plot after optimization.
        self.flag_plot = 0

        # Flag for the choice of the cost functional: 1) Infected, 2) Deceased, 3) Hospitalized;
        self.flag_cost = 3 

        # Flag for the choice of the age groups whose administrations are control variables:
        # 1) All, 2) {[60, 79], [80+]}, 3) {[0-19], [20-39], [80+]}, 4) {[40-59], [60-79]}.
        self.flag_ages = 1 

        # Flag for the test case increasing variant severity:
        # 0) 1X, 1) 2X, 2) 5X, 3) 15X, 4) 20X, 5) 30X, 6) 40X.
        self.flag_severe_factor = 0
        
        # Flag for the test case increasing variant transmission:
        # 0) 1X, 1) 1.2X, 2) 1.4X, 3) 1.6X, 4) 1.8X.
        self.flag_trans_factor = 0

        self.minimize_name = ''
        self.P = np.zeros((Nc, t_list[-1]+1, Pop.size)).squeeze() 
        self.t_contr = int(len(t_list)/self.period_U)
        self.n_controls = 3 * self.Ns_opt * self.t_contr
        self.grad_US1_d = np.zeros((6, len(t_list)))
        self.grad_US2_d = np.zeros((6, len(t_list)))
        self.grad_UR_d = np.zeros((6, len(t_list)))
        self.grad_US1 = np.zeros((6, self.t_contr))
        self.grad_US2 = np.zeros((6, self.t_contr))
        self.grad_UR = np.zeros((6, self.t_contr))
        self.I = np.zeros((6, len(t_list)))
        self.gamma = 1/self.t_rec
        self.days_del_profile = int(self.t_rec)

        self.consegne = 3e6 * self.Ns_opt/3 * np.ones(len(t_list)+1)  
        self.max_it = 20 
        self.max_elaps_time = 42
        self.min_elaps_time = 21
        self.capability_vax = 2.1e6 *  np.ones(t_list[-1] +1)
        
        # Flag for different initial policies: 1) National, 2) Homogeneous, 3) IFR-dependent, 4) Square wave.
        self.different_IC = 1

        # Flag on the kind of constraint: 0) National, 1) Homogeneous. 
        self.homo = 0

        self.delay_time = 3
        self.severity_by_age = np.array([0.181, 0.224, 0.305, 0.355, 0.646])
        
        self.stepsOC = 2000
        self.tol = 5e-4
        self.learning_rate = 0.3 
        self.err_min = 1 + self.tol
        self.history_oc = list()
        
        self.out_type = out_type
        self.out_path = out_path

        self.initialize_compartments()

    def initialize_compartments(self):
        
        if self.flag_cost == 1:
            self.minimize_name = 'Infected'
            print('Minimizing infected...')
        elif self.flag_cost == 2:
            self.minimize_name = 'Deceased'
            print('Minimizing deceased...')
        elif self.flag_cost == 3:
            self.minimize_name = 'Hospitalized'
            print('Minimizing hospitalized...')
        else:
            print('ERROR FLAG COST.')
            self.minimize_name = ''

        if self.flag_ages == 1:
            self.ages_opt = [0,1,2,3,4]
            self.ages_opt_fixed = []
            self.Ns_opt = 5

        elif self.flag_ages == 4:
            self.ages_opt = [2,3]
            self.ages_opt_fixed = [0,1,4]
            self.Ns_opt = 2

        elif self.flag_ages == 3:
            self.ages_opt = [0,1,4]
            self.ages_opt_fixed = [2,3]
            self.Ns_opt = 3

        elif self.flag_ages == 2:
            self.ages_opt = [3,4]
            self.ages_opt_fixed = [0,1,2]
            self.Ns_opt = 2

        else:
            print('ERROR FLAG AGES.')
            self.Ns_opt = 0
        
        # Initial condition on the 12th February 2021
        self.Y0 = np.array([[1.03593550e+07, 1.20218890e+07, 1.69676460e+07, 1.25935380e+07, 3.34964325e+06, 6.68320875e+05],\
                            [1.92317500e+04, 7.42610078e+04, 1.02669047e+05, 6.16688516e+04, 2.17635137e+04, 4.69438184e+03], \
                            [2.18386234e+05, 5.05087281e+05, 7.03657500e+05, 3.88334906e+05, 1.26769672e+05, 3.73827500e+04], \
                            [2.29795475e+01, 3.68488312e+02, 4.95199609e+03, 3.10931738e+04, 3.43332969e+04, 1.86817676e+04], \
                            [2.51187576e+02, 7.35470557e+03, 8.80480176e+03, 4.92116895e+03, 2.44722227e+04, 6.67090332e+03], \
                            [1.36652295e+03, 3.30053000e+05, 5.63691875e+05, 2.53181578e+05, 7.11771641e+04, 5.57923945e+04]])

        nrows = self.Ns
        ncols = len(self.t_list)
        ncolreduct = int(ncols / self.period_U)
        I = np.zeros((ncols, ncolreduct))
        for i in range(ncolreduct):
            I[i*self.period_U:(i+1)*self.period_U, i] = 1
        self.I = I
        if self.by_age:

            US1 = np.zeros((6, len(self.t_list)))
            US1[0] = self.US1[0]
            US1[1] = np.sum(self.US1[1:3], axis = 0)
            US1[2] = np.sum(self.US1[3:5], axis = 0)
            US1[3] = np.sum(self.US1[5:7], axis = 0)
            US1[4] = self.US1[7]
            US1[5] = self.US1[8]
            self.US1 = US1 @ I 
            
            US2 = np.zeros((6, len(self.t_list)))
            US2[0] = self.US2[0]
            US2[1] = np.sum(self.US2[1:3], axis = 0)
            US2[2] = np.sum(self.US2[3:5], axis = 0)
            US2[3] = np.sum(self.US2[5:7], axis = 0)
            US2[4] = self.US2[7]
            US2[5] = self.US2[8]
            self.US2 = US2 @ I
            
            UR = np.zeros((6, len(self.t_list)))
            UR[0] = self.UR[0]
            UR[1] = np.sum(self.UR[1:3], axis = 0)
            UR[2] = np.sum(self.UR[3:5], axis = 0)
            UR[3] = np.sum(self.UR[5:7], axis = 0)
            UR[4] = self.UR[7]
            UR[5] = self.UR[8]
            self.UR = UR @ I
        
        self.data['Infected'] /= self.Delta_t.flatten()
        
        #Reduction to 5 age-classes.
        self.Ns = 5
        self.Y0 = np.append(self.Y0[:, :4], np.add.reduce(self.Y0[:,4:6], axis=1).reshape(6,1), axis = 1)
        self.Y = np.zeros((6, self.t_list[-1] + 1, 5))
       

        if self.homo ==1:
            self.US1 = 2.1e6/5 * np.ones((5, self.US1.shape[1]))
            self.US2 = 1.05e6/5 * np.ones((5, self.US1.shape[1]))
            self.US2[:,:self.delay_time] = np.zeros((5, self.delay_time))
        
        self.US1 = np.append(self.US1[:4,:], np.add.reduce(self.US1[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0) 
        if self.different_IC == 2:
            pop = np.append(self.Pop[:4], np.add.reduce(self.Pop[4:]))
            self.age_percentage = ( pop/ np.sum(pop)).reshape(len(pop),1)
            self.tot_doses_age = np.sum(self.US1, axis = 0).reshape(1,self.US1.shape[1])
            self.US1 = self.age_percentage @self.tot_doses_age
            self.US2 = np.append(self.US2[:4,:], np.add.reduce(self.US2[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0)
            self.US2[self.ages_opt,self.delay_time:] =self.US1[self.ages_opt,:-self.delay_time]
        else:
            self.US2 = np.append(self.US2[:4,:], np.add.reduce(self.US2[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0) 
            self.US2[self.ages_opt,self.delay_time:] =self.US1[self.ages_opt,:-self.delay_time]
        

        self.UR = np.append(self.UR[:4,:], np.add.reduce(self.UR[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0) 
        
        if self.flag_severe_factor == 0:
            self.f = np.array([0.0001, 0.0006, 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])
        elif self.flag_severe_factor == 1:
            self.f = np.array([2* 0.0001, 2* 0.0006, 2* 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])
        elif self.flag_severe_factor == 2:
            self.f = np.array([5* 0.0001, 5* 0.0006, 5* 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])
        elif self.flag_severe_factor == 3:
            self.f = np.array([15* 0.0001, 15* 0.0006, 15* 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])
        elif self.flag_severe_factor == 4:
            self.f = np.array([20* 0.0001, 20* 0.0006, 20* 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])
        elif self.flag_severe_factor == 5:
            self.f = np.array([30* 0.0001, 30* 0.0006, 30* 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])
        elif self.flag_severe_factor == 6:
            self.f = np.array([40* 0.0001, 40* 0.0006, 40* 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])
        else:
            self.f = np.array([0.0001, 0.0006, 0.0045, 0.0231, (0.04310*self.Pop[4] + 0.1737*self.Pop[5])/(self.Pop[4] + self.Pop[5])])

        if self.different_IC == 3:
            ifr = np.array(self.f / np.sum(self.f)).reshape(len(self.f),1)
            self.tot_doses_age = np.sum(self.US1, axis = 0).reshape(1,self.US1.shape[1])
            self.US1 = ifr @self.tot_doses_age
            self.US2 = np.append(self.US2[:4,:], np.add.reduce(self.US2[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0)
            self.US2[self.ages_opt,self.delay_time:] =self.US1[self.ages_opt,:-self.delay_time]
        
        elif self.different_IC == 4:
            self.tot_doses_age = np.sum(self.US1, axis = 0).reshape(1, self.US1.shape[1]) 
            pop = np.append(self.Pop[:4], np.add.reduce(self.Pop[4:]))
            self.age_percentage = ( pop/ np.sum(pop)).reshape(len(pop),1)
            self.US1 = self.age_percentage @self.tot_doses_age
            cols = [3,4,5,9,10,11,12]
            self.US1[:, cols] = 0
            self.US2 = np.append(self.US2[:4,:], np.add.reduce(self.US2[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0)
            self.US2[self.ages_opt,self.delay_time:] =self.US1[self.ages_opt,:-self.delay_time]
            self.US2[:, :3] = 0
            self.US2[:, 3:] = self.US1[:, :-3]
        
        delta_vec = (self.Delta_t[:,4] * self.Pop[4] + self.Delta_t[:,5] * self.Pop[5])/(self.Pop[4] + self.Pop[5])
        self.Delta_t = np.append(self.Delta_t[:,:4], delta_vec.reshape(len(delta_vec),1), axis = 1)
        self.Pop = np.append(self.Pop[:4], np.add.reduce(self.Pop[4:]))
        
        if self.flag_trans_factor == 0:
            self.r_a = np.array([0.33, 1, 1, 1, 1.47])
        elif self.flag_trans_factor == 1:
            self.r_a = 1.2* np.array([0.33, 1, 1, 1, 1.47])
        elif self.flag_trans_factor == 2:
            self.r_a = 1.4* np.array([0.33, 1, 1, 1, 1.47])
        elif self.flag_trans_factor == 3:
            self.r_a = 1.6* np.array([0.33, 1, 1, 1, 1.47])
        elif self.flag_trans_factor == 4:
            self.r_a = 1.8* np.array([0.33, 1, 1, 1, 1.47])
        else:
            self.r_a = np.array([0.33, 1, 1, 1, 1.47])
        
        self.Sfrac = np.zeros((self.t_list[-1]+1, self.Ns)).squeeze()
        self.DO = np.array([[1.,         0.21075059, 0.27299633, 0.09077079, 0.04618211],\
                   [0.16120985, 0.6254323,  0.35268119, 0.12717198, 0.04852181],\
                   [0.32821309, 0.34625857, 0.45436856, 0.16735226, 0.07623629],\
                   [0.16315187, 0.24007311, 0.28991472, 0.24107584, 0.09708567],\
                   [0.20141032, 0.20209076, 0.25652598, 0.22250397, 0.08641591]])
        Nc = 6 
        self.P = np.zeros((Nc, self.t_list[-1]+1, self.Ns)).squeeze() 
        self.grad_US1_d = np.zeros((self.Ns, len(self.t_list)))
        self.grad_US2_d = np.zeros((self.Ns, len(self.t_list)))
        self.grad_UR_d = np.zeros((self.Ns, len(self.t_list)))
        self.grad_US1 = np.zeros((self.Ns, self.t_contr))
        self.grad_US2 = np.zeros((self.Ns, self.t_contr))
        self.grad_UR = np.zeros((self.Ns, self.t_contr))
        
        self.codes = ['0-19', '20-39', '40-59', '60-79', '80+']
        
        tot_doses_period = (self.US1 + self.US2 + self.UR) 
        tot_doses_susc = (self.US1 + self.US2)  
        self.consegne = np.sum(tot_doses_period, axis = 0)
        self.dosi_1_fixed = np.sum(self.US1[self.ages_opt_fixed, :], axis = 0)
        self.dosi_2_fixed = np.sum(self.US2[self.ages_opt_fixed, :], axis = 0)
        self.dosi_R_all   = np.sum(self.UR[:,:], axis = 0)
        
        self.sigma_bar = 0.31 #average infections severe fraction
        self.muH = 0.276 #average mortality for hospitalized
        self.hospitalized_fraction = 1#self.f / (self.sigma_bar * self.muH)
        
        return
    
    def plot_beta(self, t):
        return 

    def fatality_fun(self, t):
        if t <= 15:
            return ((self.Y0[0] + self.sigma1 * self.theta1 * self.Y0[4] + self.sigma2 * self.theta2 * self.Y0[5])/ \
                    (self.Y0[0] + self.sigma1 * self.Y0[4] + self.sigma2 * self.Y0[5]))
        else:
            return ((self.Y[0, t-15] + self.sigma1 * self.theta1 * self.Y[4, t-15] + self.sigma2 * self.theta2 * self.Y[5, t-15])/ \
                    (self.Y[0, t-15] + self.sigma1 * self.Y[4, t-15] + self.sigma2 * self.Y[5, t-15]))
    
    def model(self, t, y0, beta, US1, US2, UR):
        t_int = jnp.floor(t).astype(int) 
        
        S, I, R, D, V, W = y0
        if self.by_age == 0:
            US1 = self.US1[t_int]
            US2 = self.US2[t_int]
            UR = self.UR[t_int]
        
        gamma = 1 / self.t_rec
         
        muR = 1/120
        
        dSdt = - beta * self.r_a * S * jnp.dot(self.DO, I / self.Pop) - jnp.clip(US1 * self.Sfrac[t_int],0, S) + muR*R
        dIdt = beta * self.r_a * (S + self.sigma1 * V + self.sigma2 * W) * jnp.dot(self.DO, I / self.Pop) - gamma * I 
        dRdt = gamma * (1 - self.f * self.fatality_fun(t_int)) * I - jnp.clip(UR,0,  R) - muR * R
        dDdt = gamma * self.f *self.fatality_fun(t_int)* I
        dVdt = - beta * self.r_a * self.sigma1 * V * jnp.dot(self.DO, I / self.Pop) + jnp.clip(US1 * self.Sfrac[t_int],0, S) - jnp.clip(US2,0,V)
        dWdt = - beta * self.r_a * self.sigma2 * W * jnp.dot(self.DO, I / self.Pop) + jnp.clip(US2,0,V) + jnp.clip(UR,0, R)
        
        return jnp.vstack((dSdt, dIdt, dRdt, dDdt, dVdt, dWdt)).squeeze()

    def grad_U(self):
        B_hat   = np.array([-1, 0, 0, 0, 1, 0])
        B_tilde = np.array([0, 0, 0, 0, -1, 1])
        for t in range(int(self.t_list[-1]/self.period_U)+1):
            if t <= self.t_list[-1] - self.delay_time:
                self.grad_US1[self.ages_opt, t] =(  self.P[:,t,self.ages_opt].T @ B_hat + self.P[:, t + self.delay_time, self.ages_opt].T @ B_tilde)
            else:
                self.grad_US1[self.ages_opt, t] = ( self.P[:,t,self.ages_opt].T @ B_hat)
        return

    def model_MCMC(self, params, data):
        t_list = data.xdata[0].squeeze()
        self.t_list = t_list.copy()
        self.params.params[self.params.getMask()] = params[:-2*self.Ns-1]
        self.params.forecast(self.params.dataEnd,self.t_list[-1],0,None)
        
        Y0 = data.ydata[0].squeeze()
        if self.by_age:
            Y0 = np.reshape(Y0, (self.Ns,self.Ns))
        self.Y0 = Y0.copy()
        self.Y0[1] *= params[-2*self.Ns:-self.Ns]
        self.Y0[2] *= params[-self.Ns:]
        self.Y0[0] = self.Pop - self.Y0[1:].sum(axis=0)
        self.t_rec = params[-2*self.Ns-1: -2*self.Ns] 

        self.solve()
        
        forecast = data.user_defined_object[0][0]

        results = self.Y[:,self.t_list].copy()
        
        return results.transpose()
    
    def solve_ham(self, ham_grad, p0):
        nsteps = int(self.t_list[-1] + 1)
        
        self.P[:, nsteps-1] = p0
        US1 = self.US1[:, -1]/7
        US2 = self.US2[:, -1]/7
        US2[self.ages_opt] = self.US1[self.ages_opt, -1 - self.delay_time]/7
        UR = self.UR[:, -1]/7
        beta = self.params.params_time[nsteps-1]
        beta = np.delete(beta, -1) 
        for i in range(nsteps):
            t = self.t_list[-1] - i

            p0 = self.P[:, t ]
            US1_old = self.US1[:, int(np.floor((t-1)/self.period_U ))]/self.period_U
            US2_old = self.US1[:, int(np.floor((t-1)/self.period_U ))]/self.period_U
            if int(np.floor((t-1)/self.period_U )) >= self.delay_time:
                US2_old[self.ages_opt] = self.US1[self.ages_opt, int(np.floor((t-1)/self.period_U )) - self.delay_time]/self.period_U
            else:
                US2_old[self.ages_opt] = self.US2[self.ages_opt, int(np.floor((t-1)/self.period_U))]/self.period_U 
            UR_old = self.UR[:, int(np.floor((t-1)/self.period_U ))]/self.period_U
            
            beta_old = self.params.params_time[t-1]
            beta_old = np.delete(beta_old, -1)

            k1 = -ham_grad( t, self.Y[:, t ], US1, US2, UR, beta, p0)
            k2 = -ham_grad( t - 0.5, self.Y[:, t ], US1, US2, UR, beta, p0 - 0.5 * k1)
            k3 = -ham_grad( t - 0.5, self.Y[:, t ], US1, US2, UR, beta, p0 - 0.5 * k2)
            k4 = -ham_grad( t - 1, self.Y[:, t -1], US1_old, US2_old, UR_old, beta_old, p0 - k3)
            
            self.P[:, t  -1 ] = p0 - (k1 + 2*k2 + 2*k3 + k4)/6
            
            US1 = US1_old.copy()
            US2 = US2_old.copy()
            UR = UR_old.copy()
            beta = beta_old
            
        return

    def solve(self):
        t_start = int(self.t_list[0])
        self.params.compute_param_over_time(int(self.t_list[-1]))
        self.Y[:,t_start] = self.Y0
        if self.by_age == 0:
            self.Sfrac[0] = self.Y0[0] / (self.Y0[0] + (1 - self.Delta_t.values[0]) * self.Y0[2])
        else:
            self.Sfrac[0] = np.clip(self.Y0[0] / (self.Y0[0] + (1 - self.Delta_t[0,:].T) * self.Y0[2]),0,1)
        for i,t in enumerate(self.t_list[:-1]):
            y0 = self.Y[:,i+t_start]
            k = int(np.floor(t/self.period_U ))
            k_1 = int(np.floor((t+1)/self.period_U))
            beta =  self.params.params_time[t]
            beta = np.delete(beta, -1)
            US2 = self.US2[:, k].copy()
            US2[self.ages_opt] = self.US1[self.ages_opt,k-self.delay_time].copy() 
            US2_1 = self.US2[:, k_1].copy()
            US2_1[self.ages_opt] = self.US1[self.ages_opt,k_1-self.delay_time].copy()

            if k >= self.delay_time: 
                k1=self.model(t      , y0     , beta, self.US1[:, k]/self.period_U, US2/self.period_U, self.UR[:, k]/self.period_U)
                k2=self.model(t+0.5, y0+0.5*k1, beta, self.US1[:, k]/self.period_U, US2/self.period_U, self.UR[:, k]/self.period_U)
                k3=self.model(t+0.5, y0+0.5*k2, beta, self.US1[:, k]/self.period_U, US2/self.period_U, self.UR[:, k]/self.period_U)
                k4=self.model(t+1  , y0+    k3, np.delete(self.params.params_time[t+1],-1), self.US1[:, k_1]/self.period_U, self.US1[:, k_1 - self.delay_time]/self.period_U, self.UR[:, k_1]/self.period_U) 
            else:
                k1=self.model(t      , y0     , beta, self.US1[:, k]/self.period_U, self.US2[:, k]/self.period_U, self.UR[:, k]/self.period_U)
                k2=self.model(t+0.5, y0+0.5*k1, beta, self.US1[:, k]/self.period_U, self.US2[:, k]/self.period_U, self.UR[:, k]/self.period_U)
                k3=self.model(t+0.5, y0+0.5*k2, beta, self.US1[:, k]/self.period_U, self.US2[:, k]/self.period_U, self.UR[:, k]/self.period_U)
                if k_1 == self.delay_time:
                    k4=self.model(t+1  , y0+    k3, np.delete(self.params.params_time[t+1], -1), self.US1[:, k_1]/self.period_U, US2_1/self.period_U, self.UR[:, k_1]/self.period_U) 
                else:
                    k4=self.model(t+1  , y0+    k3, np.delete(self.params.params_time[t+1], -1), self.US1[:, k_1]/self.period_U, self.US2[:, k_1]/self.period_U, self.UR[:, k_1]/self.period_U) 
            self.Y[:,t_start+i+1] = y0 + (k1+2*(k2+k3)+k4)/6.0
            if self.by_age == 0:
                self.Sfrac[i+1] = self.Y[0,t_start +  i+1] / (self.Y[0,t_start + i+1] + (1 - self.Delta_t.values[i+1]) * self.Y[2,t_start+ i+1])
            else:
                if i < self.Delta_t.shape[0]-1:
                    self.Sfrac[i+1] = np.clip(y0[0] / (y0[0] + (1 - self.Delta_t[i+1,:].T) * y0[2]), 0, 1)
                else:
                    self.Sfrac[i+1] =1 
        return
    
    def lagrangian(self, t, y, US1, US2, UR):
        ages_opt = np.array(self.ages_opt)
        if self.minimize_name == 'Infected':
            return (y[1,: ]**2).sum()
        
        elif self.minimize_name == 'Hospitalized':
            frac = np.ones(5)
            frac =((y[0, :] + self.theta1 * self.sigma1 * y[4, :] + self.theta2 * self.sigma2 * y[5, :])/(y[0, :] + self.sigma1 * y[4,:] + self.sigma2 * y[5, :]))
            return ((self.severity_by_age * self.hospitalized_fraction * y[1,:] * frac)**2).sum()
        
        elif self.minimize_name == 'Deceased':
            return (y[3, :]**2).sum()# + 1e-5 * (US1[ages_opt]**2 + UR[ages_opt]**2).sum()
        
        elif self.minimize_name == 'Both':
            return 1e-5*(y[1, self.ages_opt]**2).sum() + (y[3, self.ages_opt]**2).sum()
        
        return 

    def hamiltonian(self, t, y, US1, US2, UR, beta, p0):
        return self.lagrangian(t, y, US1, US2, UR) + jnp.sum(jnp.sum(p0 * self.model(t,y,beta, US1, US2, UR)))

    def lagrangian_2(self, y, US1, UR):
        ages_opt = np.array(self.ages_opt)
        if self.minimize_name == 'Infected':
            return ((y[1,:,self.ages_opt ]**2).sum()).sum()

        elif self.minimize_name == 'Deceased':
            return ((y[3,:, self.ages_opt]**2).sum()).sum()
        
        elif self.minimize_name == 'Both':
            return 1e-5*(y[1, self.ages_opt]**2).sum() + (y[3, self.ages_opt]**2).sum()
        
        return 

    def hamiltonian_2(self, y, US1, UR, p0):
        sum_model = 0 
        for t in range(self.t_list[-1]):
            week = int(t/7)
            beta = self.params.params_time[t]
            beta = np.delete(beta, -1)

            if week >= self.delay_time:
                sum_model += jnp.sum(jnp.sum(p0[:, t, :]* self.model(t,y[:, t, :],beta, US1[:,week]/self.period_U, US1[:,week-self.delay_time]/self.period_U , UR[:,week]/self.period_U)))
            
            else:
                sum_model += jnp.sum(jnp.sum(p0[:, t, :]* self.model(t,y[:, t, :],beta, US1[:,week]/self.period_U, self.US2[:,week]/self.period_U, UR[:,week]/self.period_U)))
        
        return self.lagrangian_2(y, US1, UR) + sum_model

    def cost_functional_2(self):
        return self.lagrangian_2(self.Y, self.US1, self.UR)
    
    def cost_functional(self):
        cost_final = 0.0
        cost_interval = 0.0
        self.US2[self.ages_opt, self.delay_time:] = self.US1[self.ages_opt, :-self.delay_time]
        if self.minimize_name == 'Both':
            cost_final += (self.Y[3,-1, :]**2).sum()
        
        for i in range(self.t_list[-1] + 1):    
            beta = self.params.params_time[i]
            beta = np.delete(beta, -1) 
            k = int(np.floor(i / self.period_U ))
            if i == 0 or i == self.t_list[-1]:
                cost_interval += self.lagrangian(i,self.Y[:,i,:], self.US1[:,k]/self.period_U, self.US2[:,k]/self.period_U, self.UR[:,k]/self.period_U)/2
            else:
                cost_interval += self.lagrangian(i,self.Y[:, i,:], self.US1[:,k]/self.period_U, self.US2[:,k]/self.period_U, self.UR[:,k]/self.period_U)
        return (cost_interval + cost_final)

    def projection_simplex_sort(self, v, z=1):
        if ((np.sum(v) - z <= 0) & (np.all(v >= np.zeros(len(v))))):
            return v
        
        if z <= 0:
            return np.zeros(v.shape)
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = (u - cssv / ind) > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    def projection(self):
        for k in range(int(self.t_list[-1]/self.period_U)):
            if k >= self.delay_time:
                
                if (self.US1[self.ages_opt, k].sum() >= self.consegne[k] -self.dosi_1_fixed[k]-self.dosi_2_fixed[k]-self.dosi_R_all[k]-np.sum(self.US1[self.ages_opt, k-self.delay_time])):
                    vec = self.US1[self.ages_opt,k]
                    vec = self.projection_simplex_sort(vec, self.consegne[k] -self.dosi_1_fixed[k]-self.dosi_2_fixed[k]-self.dosi_R_all[k]-np.sum(self.US1[self.ages_opt, k-self.delay_time]))
                    self.US1[self.ages_opt, k] = vec               
                    #print('Projection at time',k,'...')

                else:
                    pass
            
            else:
            
                if (self.US1[self.ages_opt, k].sum() >= self.consegne[k] -self.dosi_1_fixed[k]-self.dosi_2_fixed[k]-self.dosi_R_all[k]-np.sum(self.US2[self.ages_opt, k])):
                    vec = self.US1[self.ages_opt,k]
                    vec = self.projection_simplex_sort(vec, self.consegne[k] -self.dosi_1_fixed[k]-self.dosi_2_fixed[k]-self.dosi_R_all[k]-np.sum(self.US2[self.ages_opt, k]))
                    self.US1[self.ages_opt, k] = vec
                    #print('Projection at time',k,'with second doses of the previous period...')
                
                else:
                    pass
        return 

    def optimal_control(self):
        
        self.projection()
        init_US1 = self.US1.copy()
        init_US2 = self.US2.copy()
        init_UR = self.UR.copy()
       
        self.solve()
        D = self.Y[3,:,self.ages_opt].copy()
        I = self.Y[1,:,self.ages_opt].copy()
        
        grad_x = jax.grad(self.hamiltonian,1) #AD
        
        grad_US1 = jax.grad(self.hamiltonian_2, 1)

        grad_x_jit = grad_x #jax.jit(grad_x)
        grad_US1_jit = grad_US1 #jax.jit(grad_US1)

        sum_gradient = 0.0
        sum_old = 1.0
        ep = 1

        print('Gradient Loop...')
        print('--------------------------------------------------------------')

        self.solve()
        
        while (ep <= self.stepsOC and self.err_min > self.tol):
            p0 = np.zeros((self.Nc, self.Ns))
            sum_gradient = 0.0

            if self.minimize_name == 'Both' or self.minimize_name == 'Deceased':
                p0[3,:] = 2* self.Y[3,-1,:] 
            if self.minimize_name == 'Hospitalized':
                p0[0,:] = self.severity_by_age * self.hospitalized_fraction * self.Y[1,-1,:] * (((1- self.theta1) * self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + (1-self.theta2) * self.sigma2 * self.Y[5, -1-self.days_del_profile, :])/(self.Y[0, -1-self.days_del_profile, :] + self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.sigma2 * self.Y[5, -1-self.days_del_profile, :])**2)
                p0[1,:] = self.severity_by_age * self.hospitalized_fraction * ((self.Y[0, -1-self.days_del_profile, :] + self.theta1 * self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.theta2 * self.sigma2 * self.Y[5, -1-self.days_del_profile, :])/(self.Y[0, -1-self.days_del_profile, :] + self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.sigma2 * self.Y[5, -1-self.days_del_profile, :]))
                p0[4,:] = self.severity_by_age * self.hospitalized_fraction * self.Y[1,-1,:] *((self.theta1 * self.sigma1 *(self.Y[0, -1-self.days_del_profile, :] + self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.sigma2 * self.Y[5, -1-self.days_del_profile, :]) - self.sigma1 * (self.Y[0, -1-self.days_del_profile, :] + self.theta1 * self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.theta2 * self.sigma2 * self.Y[5, -1-self.days_del_profile, :]))/((self.Y[0, -1-self.days_del_profile, :] + self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.sigma2 * self.Y[5, -1-self.days_del_profile, :])**2)) 
                p0[5,:] = self.severity_by_age * self.hospitalized_fraction * self.Y[1,-1,:] *((self.theta2 * self.sigma2 *(self.Y[0, -1-self.days_del_profile, :] + self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.sigma2 * self.Y[5, -1-self.days_del_profile, :]) - self.sigma2 * (self.Y[0, -1-self.days_del_profile, :] + self.theta1 * self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.theta2 * self.sigma2 * self.Y[5, -1-self.days_del_profile, :]))/((self.Y[0, -1-self.days_del_profile, :] + self.sigma1 * self.Y[4, -1-self.days_del_profile,:] + self.sigma2 * self.Y[5, -1-self.days_del_profile, :])**2)) 
                p0 *= 2
            else:
                p0[1,:] = 2* self.Y[1,-1,:] 
            
            self.P[:,-1,:] = p0.copy()
            
            # Solution of the state problem. 
            self.projection()
            self.solve()

            # Solution of the adjoint problem.
            self.solve_ham(grad_x_jit, p0)

            US1_old = self.US1.copy()
            
            # Computing the gradient.
            self.grad_U()

            b = 0.5# 0.1
            J_old = self.cost_functional()
            cost = list()
            
            self.US1 = np.array(US1_old - self.grad_US1)
            
            self.projection()
            US1_1 = self.US1.copy()
            
            self.US1 = US1_old.copy()
            
            k = 1
            Y_old = self.Y
            while (k <= 5): #Armijo condition
                self.US1 = np.array(US1_old - b * self.learning_rate *  self.grad_US1)
                
                self.projection()
                self.solve()
                Y = self.Y.copy()
                Y_old = Y.copy()
                
                J_new = self.cost_functional()
                
                if (J_new <= (J_old - self.learning_rate / b * (np.sum(np.sum((self.US1[self.ages_opt,:] - US1_old[self.ages_opt,:])**2))))):
                    k = 6 
                
                else:
                    k = k+1
                    b = b /2 
            
            J_new = self.cost_functional()
            self.history_oc.append(J_new)
            self.err_min = (np.sum(np.sum((self.US1 - US1_1)**2)))  
            
            ep += 1
            
            print('First check: ', ep <= self.stepsOC)
            print('Second check: ', self.err_min > self.tol)
            print('Iteration: ', ep)
            print('Cost functional: ', self.history_oc[-1])
            print('--------------------------------------------------------------')

        D_new = self.Y[3,:, self.ages_opt].copy()
        I_new = self.Y[1,:, self.ages_opt].copy()
            
        self.flag_plot == 0:
            fig2, ax2 = plt.subplots(1,1)

            ax2.plot(np.sum(I, axis =0), '*')
            ax2.plot(np.sum(I_new, axis = 0))
            ax2.set_title('Infected')
            ax2.legend(['Old', 'New'])
            plt.show()
            plt.savefig(self.out_path+'/img/Infected_ages_opt.png')

            fig3, ax3 = plt.subplots(1,1)
            ax3.plot(np.sum(D, axis =0), '*')
            ax3.plot(np.sum(D_new, axis = 0))
            ax3.set_title('Deceased')
            ax3.legend(['Old', 'New'])
            plt.show()
            plt.savefig(self.out_path+'/img/Deceased_ages_opt.png')
            
            fig4, ax4 = plt.subplots(1,1)
            ax4.plot(self.history_oc)
            ax4.set_title('Cost')
            plt.grid()
            plt.savefig(self.out_path+'/img/Cost.png')
            plt.show()
        
        print('End Loop...')
        print('--------------------------------------------------------------')
        
        self.US2[self.ages_opt,self.delay_time:] = self.US1[self.ages_opt,:-self.delay_time]
        return
                                
              
    def estimate(self):
        params0 = pm.maskParams( self.params.get() , self.params.getMask() )
        lower_b = pm.maskParams( self.params.getLowerBounds() , self.params.getMask() )
        upper_b = pm.maskParams( self.params.getUpperBounds() , self.params.getMask() )
        bounds = Bounds( lower_b, upper_b )
        
        #local mimimization
        result = minimize_parallel(self.error_LS, params0, bounds=bounds,\
                options={'ftol': 1e-15, 'maxfun':1000, 'maxiter':1000,'iprint':1})
        print('###########################################')
        print(result)
        print('###########################################')
        # assign estimated parameters to model parameters
        self.params.params[self.params.getMask()] = result.x
        return

    def error(self, params0):
        self.params.params[self.params.getMask()] = params0
        self.solve()
        S, I, R, D, V, W = self.Y[:,self.t_list]

        error_D = np.absolute(D.flatten() - self.data['Deceased'].values) 
        one = np.ones(self.Ns)
        weight = np.ones(self.Ns)
        # Modify this if you want to weight more specific time steps.
        weightsD = weight
        i = 0
        
        for age in ['0-19', '20-39', '40-59', '60-79', '80-89', '90+']:
            weightsD[i] = np.clip(weightsD[i]/max(error_D[i::self.Ns]), 1, None)
            i+=1
        
        error_Dtot = np.sum(error_D.reshape((self.Ns, self.t_list[-1]+1)), axis = 1)
        weight_total_D = np.ones(error_Dtot.shape[0])
        weight_total_D = weightsD /  np.array([0.01, 0.01, 0.05, 1, 1,1])
        errorL2 = np.sum((error_Dtot*weight_total_D)**2)
        
        return errorL2
    
    def error_LS(self, params):
        error = self.error(params)
        return np.sqrt(error)

    def error_MCMC(self, params, data):
        Y0 = data.ydata[0].squeeze()
        
        self.Y0 = Y0.copy()
        self.Y0[1] *= params[-2*self.Ns:-self.Ns]
        self.Y0[2] *= params[-self.Ns:]
        self.Y0[0] = self.Pop - self.Y0[1:].sum(axis=0)
        self.t_rec = params[-2*self.Ns-1:-2*self.Ns]
        return self.error(params[:-2*self.Ns - 1])

    def computeRt(self): 
        self.Rt = np.zeros(len(self.t_list)) 
        gamma = 1/self.t_rec
        NGM_matrix = np.zeros((5,5))
        ra = self.r_a.reshape((1,5))
        Ri_matrix = np.repeat(ra, 5, axis = 0)
        states_matrix = np.zeros((5,5))
        
        for i,t in enumerate(self.t_list):
            t_int = int(t)
            beta = self.params.params_time[int(t)][:-1]
            states_matrix = np.repeat(((self.Y[0, t_int, :] + self.sigma1 * self.Y[4, t_int, :] + self.sigma2 * self.Y[5, t_int, :])/self.Pop).reshape((1,5)), 5, axis = 0)
            R0_tmp = beta / gamma
            NGM_matrix = R0_tmp *Ri_matrix*states_matrix*self.DO
            self.Rt[t_int] = max(abs(np.linalg.eigvals(NGM_matrix)))
        
        np.savetxt(self.out_path+'/Rt.csv', self.Rt, delimiter=',')
        return

    def save(self):
        print('Reorganizing and saving results...')
        # Sum undetected from base and variant
        self.Y = self.Y[:,self.t_list]

        Nc = self.Y.shape[0]
        codes = np.tile(self.codes, len(self.t_list))
        times = np.repeat(self.t_list, len(self.codes))
        
        dates = [self.DPC_start + pd.Timedelta(t, 'days') for t in times]

        results = np.zeros((Nc+3, len(times)), dtype='O')
        US1_df = pd.DataFrame(self.US1,columns = [self.DPC_start + pd.Timedelta(7*t,'days') for t in range(self.t_contr)])
        US2_df = pd.DataFrame(self.US2,columns = [self.DPC_start + pd.Timedelta(7*t,'days') for t in range(self.t_contr)])
        UR_df = pd.DataFrame(self.UR,columns = [self.DPC_start + pd.Timedelta(7*t,'days') for t in range(self.t_contr)])
        history_conv_df = pd.DataFrame(self.history_oc)

        results[:3] = codes, dates, times
        results[3:3+Nc] = self.Y.reshape(Nc, len(times))
        
        Code = "Age" if self.by_age else "Geocode"
        results_df = pd.DataFrame(results.T,columns=[Code,'date','time','Suscept','Infected','Recovered', 'Deceased', 'VaccinatedFirst', 'VaccinatedSecond'])
        
        if not self.by_age:
            results_df = results_df.astype({Code: int,"time": 'float64'})
        else:
            results_df = results_df.astype({Code: str,"time": 'float64'})

        results_df = results_df.sort_values(by=[Code,'date'])
        results_df = results_df.astype(dict(zip(['Suscept','Infected','Recovered', 'Deceased', 'VaccinatedFirst', 'VaccinatedSecond'],['float64']*5)))

        outFileName = self.out_path+'/simdf.'+self.out_type
        
        if self.out_type == 'csv':
            results_df.to_csv(outFileName,index=False)
            US1_df.to_csv(self.out_path+'/US1.'+self.out_type, index=False)
            US2_df.to_csv(self.out_path+'/US2.'+self.out_type, index=False)
            UR_df.to_csv(self.out_path+'/UR.'+self.out_type, index=False)
            history_conv_df.to_csv(self.out_path+'/history_conv.'+self.out_type, index=False)

        elif self.out_type == 'h5':
            results_df.to_hdf(outFileName, key='results_df', mode='w')
            US1_df.to_hdf(self.out_path+'/US1.'+self.out_type, key='US1_df', mode='w')
            US2_df.to_hdf(self.out_path+'/US2.'+self.out_type, key='US2_df', mode='w')
            US1_df.to_hdf(self.out_path+'/UR.'+self.out_type, key='UR_df', mode='w')
            history_conv_df.to_hdf(self.out_path+'/history_conv.'+self.out_type, key='history_conv_df', mode='w')

        print('...done!')

        return

    def save_weeks_plus(self, weeks_plus):
        print('Reorganizing and saving results...')
        # Sum undetected from base and variant
        self.solve()
        self.Y = self.Y[:,self.t_list]

        Nc = self.Y.shape[0]
        codes = np.tile(self.codes, len(self.t_list))
        times = np.repeat(self.t_list, len(self.codes))
        dates = [self.DPC_start + pd.Timedelta(t, 'days') for t in times]
        
        self.t_contr += weeks_plus
        results = np.zeros((Nc+3, len(times)), dtype='O')
        US1_df = pd.DataFrame(self.US1,columns = [self.DPC_start + pd.Timedelta(7*t,'days') for t in range(self.t_contr)])
        US2_df = pd.DataFrame(self.US2,columns = [self.DPC_start + pd.Timedelta(7*t,'days') for t in range(self.t_contr)])
        UR_df = pd.DataFrame(self.UR,columns = [self.DPC_start + pd.Timedelta(7*t,'days') for t in range(self.t_contr)])
        history_conv_df = pd.DataFrame(self.history_oc)

        results[:3] = codes, dates, times
        results[3:3+Nc] = self.Y.reshape(Nc, len(times))
        
        Code = "Age" if self.by_age else "Geocode"
        results_df = pd.DataFrame(results.T,columns=[Code,'date','time','Suscept','Infected','Recovered', 'Deceased', 'VaccinatedFirst', 'VaccinatedSecond'])
        
        if not self.by_age:
            results_df = results_df.astype({Code: int,"time": 'float64'})
        else:
            results_df = results_df.astype({Code: str,"time": 'float64'})

        results_df = results_df.sort_values(by=[Code,'date'])
        results_df = results_df.astype(dict(zip(['Suscept','Infected','Recovered', 'Deceased', 'VaccinatedFirst', 'VaccinatedSecond'],['float64']*5)))

        outFileName = self.out_path+'/simdf_1_month.'+self.out_type
        
        if self.out_type == 'csv':
            results_df.to_csv(outFileName,index=False)
            US1_df.to_csv(self.out_path+'/US1_1_month.'+self.out_type, index=False)
            US2_df.to_csv(self.out_path+'/US2_1_month.'+self.out_type, index=False)
            UR_df.to_csv(self.out_path+'/UR_1_month.'+self.out_type, index=False)
            history_conv_df.to_csv(self.out_path+'/history_conv_1_month.'+self.out_type, index=False)

        elif self.out_type == 'h5':
            results_df.to_hdf(outFileName, key='results_df', mode='w')
            US1_df.to_hdf(self.out_path+'/US1_1_month.'+self.out_type, key='US1_df', mode='w')
            US2_df.to_hdf(self.out_path+'/US2_1_month.'+self.out_type, key='US2_df', mode='w')
            US1_df.to_hdf(self.out_path+'/UR_1_month.'+self.out_type, key='UR_df', mode='w')
            history_conv_df.to_hdf(self.out_path+'/history_conv_1_month.'+self.out_type, key='history_conv_df', mode='w')

        print('...done!')

        return

