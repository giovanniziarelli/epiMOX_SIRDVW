import sys
import os.path
import numpy as np
import pandas as pd
import json
from epi import loaddata as ld
from epi.convert import converter
from epiMOX_class import epiMOX
from pymcmcstat.MCMC import DataStructure
from pymcmcstat.propagation import calculate_intervals, generate_quantiles
import pymcmcstat.chain.ChainProcessing as chproc


def MCMC_postprocess(ResPath, nsample=500, burnin=None, forecast=True, scenario=None):
    InputFileName = ResPath + '/input.inp'

    if os.path.exists(InputFileName):
        DataDic = ld.readdata(InputFileName)
        epi_model = DataDic['model']
    else:
        sys.exit('Error - Input data file ' + InputFileName + ' not found. Exit.')
    MCMCpath = ResPath + '/MCMC/'	
    if os.path.exists(ResPath):
        ResultsDict = chproc.load_serial_simulation_results(MCMCpath, json_file='results_dict.json', extension='txt')
        chain = ResultsDict['chain']
        s2chain = ResultsDict['s2chain']
    else:
        sys.exit('Error - MCMC folder in ' + ResPath + ' not found. Exit.')
    DPC_start = DataDic['DPC_start']
    DPC_ndays = DataDic['DPC_end']
    country = DataDic['country']
    if country is None:
        country = 'Italia'
    if burnin is None:
        burnin = chain.shape[0]//5

    chain = chain[burnin:,:]
    model_solver = epiMOX(ResPath,scenari=scenario)
    t_list = model_solver.t_list.copy()
    Y0 = model_solver.Y0.copy()
    epi_start = pd.to_datetime('2021-02-24') 

    day_init = pd.to_datetime(DPC_start)
    day_end = pd.to_datetime(DPC_ndays)

    mask = model_solver.params.getMask()

    variant_prevalence = float(DataDic['variant_prevalence']) if 'variant_prevalence' in DataDic.keys() else 0
    if 'variant' in DataDic.keys() and DataDic['variant']:
        with open('util/variant_db.json') as variant_db:
            variants = json.load(variant_db)
        variant = variants[DataDic['variant']]
    else:
        variant = None
        variant_prevalence=0

    args = (forecast, variant, variant_prevalence)
    data=DataStructure()
    data.add_data_set(t_list, Y0, user_defined_object=args)
    
    # Calculate intervals
    interval = calculate_intervals(chain, ResultsDict, data, model_solver.model_MCMC, nsample=nsample, waitbar=True)
    print('QUI?', interval['credible'][0].shape)
    t_list = model_solver.t_list.copy()

    compartments =  [
            'Suscettibili',
            'Infetti',
            'Guariti',
            'Deceduti',
            'Vaccinati Prima Dose',
            'Vaccinati Ciclo Completo',
            'Deceduti giornalieri',
            'Positivi',
            'Tasso di letalità',
            'R*'
    ]
    n_compartments = len(compartments)
####################################################
    quantiles = np.array([0.025, 0.5, 0.975])
####################################################
    t_list = np.insert(t_list, 0, np.arange(0,t_list[0]))
    results_df = pd.DataFrame(index = pd.date_range(day_init+pd.Timedelta(t_list[0], 'days'), day_init+pd.Timedelta(t_list[-1], 'days')), columns = compartments)
    print(interval['credible'].shape)
    intervals_dict = dict(zip(['S', 'I', 'R', 'D', 'V', 'W'],[inter['credible'] for inter in interval]))
    wanted_compartments = ['S', 'I', 'R', 'D', 'V', 'W']
    n_wanted_compartments = len(wanted_compartments)
    results = np.zeros((model_solver.t_list.size, n_wanted_compartments, quantiles.size)) 
    for i,comp in enumerate(wanted_compartments):
        model_quantiles = generate_quantiles(intervals_dict[comp], quantiles)
        results[:,i,:] = model_quantiles.T

    deceduti_giornalieri = generate_quantiles(np.diff(intervals_dict['D'], axis=1, prepend=model_solver.data.Deceased.iloc[-2]), quantiles)
    positivi = generate_quantiles(intervals_dict['I'], quantiles)
    tasso_letale = generate_quantiles(intervals_dict['D']/(intervals_dict['D']+intervals_dict['R']), quantiles)
    logI = pd.DataFrame(np.log(intervals_dict['I']).T)
    r_star = generate_quantiles((logI.diff(periods=7)/7*9+1).to_numpy().T, quantiles)

    results_df.index = results_df.index.strftime('%Y-%m-%d')
    results_df.rename_axis('Date', inplace=True)
    for i,x in enumerate(quantiles):
        results_df.iloc[:,:n_wanted_compartments] = results[...,i]
        results_df['Deceduti giornalieri'] = deceduti_giornalieri[i]
        results_df['Positivi'] = positivi[i] 
        results_df['Tasso di letalità'] = tasso_letale[i] 
        results_df['R*'] = r_star[i] 
        results_df.to_json(ResPath + f'simdf_MCMC{"_"+str(x).rsplit(".")[-1]}.json', date_format='iso')

    return data, interval
	

if __name__ == '__main__':
    # Read data file
    if len(sys.argv) < 2:
       sys.exit('Error - at least the path containing the resulting test cases is needed')
    ResultsFilePath = sys.argv[1]
    if not os.path.exists(ResultsFilePath):
    	sys.exit('Error - Input reults folder ' + ResultsFilePath + ' not found. Exit.')

    if len(sys.argv) > 2:
        nsample = int(sys.argv[2])
    else:
        nsample = 500

    if len(sys.argv) > 3:
        burnin = int(sys.argv[3])
    else:
        burnin = None

    if len(sys.argv) > 4:
        forecast = bool(int(sys.argv[4]))
    else:
        forecast = True
    
    data,interval = MCMC_postprocess(ResultsFilePath, nsample, burnin, forecast, scenario = None)
