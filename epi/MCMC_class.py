import os
import numpy as np
import pandas as pd
from pymcmcstat.MCMC import MCMC
from pymcmcstat.MCMC import DataStructure
from pymcmcstat.ParallelMCMC import ParallelMCMC

def solveMCMC(testPath, model_solver, Y0, nsimu = 1e4, sigma = 0.1*3e4, parallel=False, nproc=3, nchains=3):
    os.mkdir(testPath+'/MCMC')
    mcstat = MCMC()
    params = model_solver.params 
    mask = params.getMask()
    Dim =len(Y0)
    if Dim == 8: #SEIHRDVW
        names = ['beta','f']
    elif Dim ==3: #SIR
        names = ['beta', 'gamma']
    elif params.nParams==13: #SUIHTER
        names = ['betaU','betaI','delta','omegaI','omegaH','rhoU','rhoI','rhoH','rhoT','gamma_T','gamma_I','theta_H','theta_T']
    elif Dim == 6: #SIRDVW
        #names = ['beta', 'gamma']
        names = ['beta']
    t_list = model_solver.t_list[:int(max(model_solver.data.time.values))+1]
    mcstat.data.add_data_set(t_list,Y0)
    minimum = params.getLowerBounds()
    maximum = params.getUpperBounds()
    

    if params.nSites != 1:
        #prior_s = np.array([5.80, 5.96, 5.85, 6.13, 6.13, 6.13])/3
        prior_s = 5.94
        for i in range(params.nPhases):
            if params.nParams > 1:        
                for j in range(params.nParams):
                    for k in range(params.nSites):
                        if mask[i,j,k]:
                            mcstat.parameters.add_model_parameter(
                                name=str(k)+names[j]+str(i),
                                theta0=params.get()[i,j,k],
                                minimum=0.7*params.get()[i,j,k],
                                maximum=1.3*params.get()[i,j,k])
            else:
                for k in range(params.nSites):
                    if params.constantSites[0] == 0:
                        print('Sono quii')
                        for k in range(params.nSites):
                            if mask[i,k]:
                                mcstat.parameters.add_model_parameter(
                                    name=str(k)+'beta'+str(i),
                                    theta0=params.get()[i,k],
                                    minimum=0.7*params.get()[i,k],
                                    maximum=1.3*params.get()[i,k])
                    else:
                        print('Sono qui')
                        if mask[i,0]:
                            mcstat.parameters.add_model_parameter(
                                name='beta'+str(i),
                                theta0=params.get()[i,0],
                                minimum=0.7*params.get()[i,0],
                                maximum=1.3*params.get()[i,0])
        # Uncertainty on IC -> SIR
        if Dim == 3:
            for k in range(params.nSites):
                mcstat.parameters.add_model_parameter(
                    name=str(k)+'I0',
                    theta0=1,
                    minimum=0.7,
                    maximum=1.3)
            for k in range(params.nSites):
                mcstat.parameters.add_model_parameter(
                    name=str(k)+'R0',
                    theta0=1,
                    minimum=0.7,
                    maximum=1.3)
        # Uncertainty on IC -> SEIHRDVW
        elif Dim  == 8:
            for k in range(params.nSites):
                mcstat.parameters.add_model_parameter(
                    name=str(k)+'E0',
                    theta0=1,
                    minimum=0.7,
                    maximum=1.3)
            for k in range(params.nSites):
                mcstat.parameters.add_model_parameter(
                    name=str(k)+'I0',
                    theta0=1,
                    minimum=0.7,
                    maximum=1.3)
            for k in range(params.nSites):
                mcstat.parameters.add_model_parameter(
                    name=str(k)+'H0',
                    theta0=1,
                    minimum=0.7,
                    maximum=1.3)
            for k in range(params.nSites):
                mcstat.parameters.add_model_parameter(
                    name=str(k)+'R0',
                    theta0=1,
                    minimum=0.7,
                    maximum=1.3)
        elif Dim == 6:#SIRDVW
             mcstat.parameters.add_model_parameter(
                 name='t_rec',
                 theta0=10,
                 minimum=0,
                 maximum=25, 
                 prior_mu = model_solver.t_rec,
                 prior_sigma = prior_s)
            #for k in range(params.nSites):
            #    mcstat.parameters.add_model_parameter(
            #        name=str(k)+'t_rec',
            #        theta0=13,
            #        minimum=0,
            #        maximum=25, 
            #        prior_mu = model_solver.t_rec[k],
            #        prior_sigma = prior_s[k])
             for k in range(params.nSites):
                 mcstat.parameters.add_model_parameter(
                     name=str(k)+'I0',
                     theta0=1,
                     minimum=0.7,
                     maximum=1.3)
             for k in range(params.nSites):
                 mcstat.parameters.add_model_parameter(
                     name=str(k)+'R0',
                     theta0=1,
                     minimum=0.7,
                     maximum=1.3)
    else:
        prior_s = 5.94
        for i in range(params.nPhases):
            for j in range(params.nParams):
                if mask[i,j]:
                    mcstat.parameters.add_model_parameter(
                        name=names[j]+str(i),
                        theta0=params.get()[i,j],
                        minimum=0.5*params.get()[i,j],
                        maximum=1.5*params.get()[i,j])

        if len(names) == 13:
            mcstat.parameters.add_model_parameter(
                name='omegaI_err',
                theta0=0,
                prior_mu=0,
                prior_sigma=0.10)

<<<<<<< HEAD
            mcstat.parameters.add_model_parameter(
                name='omegaH_err',
                theta0=0,
                prior_mu=0,
                prior_sigma=0.20)

            mcstat.parameters.add_model_parameter(
                name='U0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
=======
        mcstat.parameters.add_model_parameter(
            name='U0',
            theta0=1,
            minimum=0.5,
            maximum=1.5)

        mcstat.parameters.add_model_parameter(
            name='R0',
            theta0=1,
            minimum=0.5,
            maximum=1.5)
>>>>>>> ad0639aa9c2b74ca2c5c9b1a5a93b29e1cff9c73

            mcstat.parameters.add_model_parameter(
                name='R0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
        elif Dim  == 3:
            mcstat.parameters.add_model_parameter(
                name='I0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
            mcstat.parameters.add_model_parameter(
                name='R0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
        elif Dim == 8:
            mcstat.parameters.add_model_parameter(
                name='E0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
            mcstat.parameters.add_model_parameter(
                name='I0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
            mcstat.parameters.add_model_parameter(
                name='H0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
            mcstat.parameters.add_model_parameter(
                name='R0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
        elif Dim == 6:#SIRDVW
            mcstat.parameters.add_model_parameter(
                name='t_rec',
                theta0=13,
                minimum=0,
                maximum=25, 
                prior_mu = model_solver.t_rec,
                prior_sigma = prior_s)
            mcstat.parameters.add_model_parameter(
                name='I0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
            mcstat.parameters.add_model_parameter(
                name='R0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
    mcstat.simulation_options.define_simulation_options(
        nsimu=nsimu,
        updatesigma=1,
        save_to_json=True,
        save_to_txt=True,
        results_filename='results_dict.json',
        savedir=testPath+'/MCMC/')
    mcstat.model_settings.define_model_settings(
        sos_function=model_solver.error_MCMC,
        sigma2=sigma**2)
    if parallel:
        parmcstat = ParallelMCMC()
        parmcstat.setup_parallel_simulation(mcstat,num_cores=nproc,num_chain=nchains)
        parmcstat.run_parallel_simulation()
        return parmcstat

    mcstat.run_simulation()
    return mcstat
