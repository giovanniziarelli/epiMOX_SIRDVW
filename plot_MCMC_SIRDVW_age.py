import sys
import os.path
import numpy as np
import pandas as pd
import json
import pickle as pl
import datetime
import importlib
from epi import loaddata as ld
from epi import models as md
from epi import estimation as es
from epi.convert import converter
from epiMOX_class import epiMOX
from epi.MCMC import model_fun_var_new as model_fun
import matplotlib as mpl
import zipfile
import matplotlib.pyplot as plt
from pymcmcstat.MCMC import DataStructure
from pymcmcstat import mcmcplot as mcp
from pymcmcstat.propagation import calculate_intervals, plot_intervals, generate_quantiles
import pymcmcstat.chain.ChainProcessing as chproc
from collections import Counter


if __name__ == '__main__':
    # Read data file
    if len(sys.argv) < 2:
       sys.exit('Error - at least the path containing the resulting test cases is needed')
    ResultsFilePath = sys.argv[1]
    if not os.path.exists(ResultsFilePath):
    	sys.exit('Error - Input reults folder ' + ResultsFilePath + ' not found. Exit.')
    median_df = pd.read_json(ResultsFilePath + '/simdf_MCMC_5.json')
    fquantile_df = pd.read_json(ResultsFilePath + '/simdf_MCMC_025.json')
    lquantile_df = pd.read_json(ResultsFilePath + '/simdf_MCMC_975.json')
    simdf = pd.read_hdf(ResultsFilePath + '/simdf.h5')
 
    medians_age = []
    fquant_age = []
    lquant_age = []
    count_times = len(Counter(list(simdf.date)).values())
    for i in range(6):
        medians_age.append(median_df[i*count_times: (i+1)*count_times])
        fquant_age.append(fquantile_df[i*count_times : (i+1) * count_times])
        lquant_age.append(lquantile_df[i*count_times : (i+1) * count_times])

    if not os.path.exists(ResultsFilePath + '/img/'):
        os.mkdir(ResultsFilePath + '/img/')

    ## Modify here to change the starting date ##

    fileName = ResultsFilePath + '/input.inp' 
    if os.path.exists(fileName):
        DataDic = ld.readdata(fileName)
    else:
        sys.exit('Error - Input data file ' +fileName+ ' not found. Exit.')
    
    if 'save_code' in DataDic.keys():
        save_code = bool(int(DataDic['save_code']))
    else:
        save_code = True

    if save_code:
        # copy all the code in a crompessed format to the test case folder.
        with zipfile.ZipFile(ResultsFilePath + '/epiMOX.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir('.', zipf)

    model, Nc, country, param_type, param_file, Tf, dt, save_code, by_age, edges_file, \
        borders_file, map_file, mobility, mixing, estim_param, DPC_start,\
        DPC_end, data_ext_deg, ext_deg, out_type, restart_file, day_restart, only_forecast, scenario\
        = ld.parsedata(DataDic)
    epi_start = pd.to_datetime('2020-02-24')
    
    pm = importlib.import_module('epi.parameters_'+param_type)
    param_file = ResultsFilePath + '/' + param_file
    if by_age:
        sites_file = './util/Eta_Italia_sites.csv'
    else:
        sites_file = './util/Regioni_Italia_sites.csv'
    edges_file = './util/'+edges_file
    borders_file = './util/'+borders_file
    map_file = './util/'+map_file
    out_path = ResultsFilePath

    day_init = pd.to_datetime(DPC_start)
    day_end = pd.to_datetime(DPC_end)
    day_ISS_data = pd.to_datetime('2022-12-08')

    Tf_data = pd.to_datetime(Tf)
    Tf = int((Tf_data-day_init).days)
    params = pm.Params(day_init, (day_end-day_init).days) 
    params.load(param_file)
    param_matrix = np.repeat(params.params, params.lenPhases, axis = 0)
    if (params.lenPhases - params.lenInitPhase) > 0:
        param_matrix = np.delete(param_matrix, np.s_[0:(params.lenPhases - params.lenInitPhase)], 0)
    elif (params.lenPhases - params.lenInitPhase) < 0:
        aux = np.repeat(param_matrix[0,:], params.lenInitPhase +1 - params.lenPhases ,0 )
        aux = np.reshape(aux, (params.lenInitPhase+1-params.lenPhases, params.nSites))
        param_matrix = np.insert(param_matrix,0,aux, 0)
    print(param_matrix.shape)
    eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
    eData = eData[(eData["data"]>=day_init.isoformat()) & (eData["data"]<=(Tf_data+datetime.timedelta(1)).isoformat())]

    eData['data'] = [pd.to_datetime(x[:10]) for x in eData.data]
    eData = eData.reset_index(drop=True)
    eData = converter('SIRDVW', eData, country, Nc)
    eData = eData.reset_index(drop=True)
    perc = pd.read_csv(f'~/dpc-covid-data/SUIHTER/stato_clinico_SIRDVW.csv')
    perc = perc[(perc['Data']>=DPC_start) & (perc['Data']<=Tf_data.strftime("%Y-%m-%d"))] 
    Ns = 6
    eData = pd.DataFrame(np.repeat(eData.values,Ns,axis=0),columns=eData.columns)
    eData[perc.columns[2:]] = eData[perc.columns[2:]].mul(perc[perc.columns[2:]].values)
    eData['Age'] = perc['Età'].values
    eData.sort_values(by=['Age','time'])
    #############################################

    int_colors = [(1.0, 0.6, 0.6), (1.0, 0.6, 0.6), (0.6, 0.6, 1.0), (0.6, 1.0, 0.6), (1.0, 1.0, 0.6)]
    model_colors = [(1.0, 0, 0.0), (1.0, 0, 0.0), (0, 0.0, 1.0), (0.0, 1.0, 0), (1.0, 1.0, 0.0)]
    int_colors = [ (1.0, 0.6, 0.6), (0.6, 0.6, 1.0), (0.6, 1.0, 0.6), (1.0, 0.7, 0.4), (0.4, 1.0, 0.7), (0.7, 0.4, 1.0)]
    model_colors =[ (1.0, 0, 0.0), (0, 0.0, 1.0), (0.0, 1.0, 0), (0.8, 0.4, 0.0), (0.0, 0.8, 0.4), (0.4, 0.0, 0.8)]

    linetype=['--','-','-','-','-']
    linetype=['-','-','-','-','-','-']

    MCMCpath = ResultsFilePath + '/MCMC/'	
    if os.path.exists(ResultsFilePath):
        ResultsDict = chproc.load_serial_simulation_results(MCMCpath, json_file='results_dict.json', extension='txt')
        chain = ResultsDict['chain']
        s2chain = ResultsDict['s2chain']
        names = ResultsDict['names']
    else:
        sys.exit('Error - MCMC folder in ' + ResultsFilePath + ' not found. Exit.')
    list_names = [int(i*6) for i in range(21)]
    list_names_end = [int(126 + x) for x in range(13)]
    burnin = 5000
    for i in range(len(list_names)):
        chain1 = chain[burnin:, i].reshape((20000-burnin,1))
        if i == 0 or i == 9 or i == 19:         
            mcp.plot_density_panel(chain1 , names[i] )
            plt.savefig(ResultsFilePath + 'img/' +'beta'+str(i)+'.png')
            plt.show()

    mcp.plot_density_panel(chain[burnin:, 126].reshape((20000-burnin,1)))
    plt.savefig(ResultsFilePath + 'img/' +'t_rec.png')
    plt.show()
    
    Infected = simdf.Infected
    Rec = simdf.Recovered
    Dec = simdf.Deceased
    Vaccinated1 = simdf.VaccinatedFirst
    Vaccinated2 = simdf.VaccinatedSecond
   ##################################
    titoli = ['Isolated','Deceased']
    
    date = eData[eData.Age == '0-19'].data
    medians_inf = np.zeros(len(date))
    fquant_inf = np.zeros(len(date))
    lquant_inf = np.zeros(len(date))
    medians_dec = np.zeros(len(date))
    fquant_dec = np.zeros(len(date))
    lquant_dec = np.zeros(len(date))
    for i in range(6):
        medians_inf += np.array(medians_age[i].Positivi)
        fquant_inf += np.array(fquant_age[i].Positivi)
        lquant_inf += np.array(lquant_age[i].Positivi)

        medians_dec += np.array(medians_age[i].Deceduti)
        fquant_dec += np.array(fquant_age[i].Deceduti)
        lquant_dec += np.array(lquant_age[i].Deceduti)

    
    fig,axes1 = plt.subplots(1,2, figsize = (11, 4))
    baseline = 0.0 * np.ones(len(date))
    ages = ['0-19', '20-39', '40-59', '60-79', '80-89', '90+']
    datestr = [day_init + datetime.timedelta(k) for k in range(len(date))]

    colors = plt.cm.cool(np.linspace(0, 1, 5))
    codes = ['0-19', '20-39', '40-59', '60-79', '80+']
    for k in range(5):
        if k == 4:
            perc = (np.array(medians_age[k].Positivi)+np.array(medians_age[k+1].Positivi)) / medians_inf
        else:
            perc = np.array(medians_age[k].Positivi) / medians_inf
        axes1[0].plot(datestr,perc + baseline, color = colors[k])
        axes1[0].fill_between(datestr, baseline, perc + baseline, color = colors[k], alpha = 0.5)
        baseline = baseline + perc
    axes1[0].set_title('Calibrated model')
    axes1[0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    axes1[0].tick_params(axis='both', which='major', labelsize=7)
    axes1[0].tick_params(axis='both', which='minor', labelsize=7)
    baseline = 0.0 * np.ones(len(date))
    for k in range(5):
        if k ==4:
            perc = (np.array(eData[eData.Age == ages[k]].Infected)+np.array(eData[eData.Age == ages[k+1]].Infected)) / np.array(eData.groupby(by = 'data').Infected.sum())
        else:
            perc = np.array(eData[eData.Age == ages[k]].Infected) / np.array(eData.groupby(by = 'data').Infected.sum())
        axes1[1].plot(datestr,perc+ baseline, color = colors[k])
        axes1[1].fill_between(date, np.array(baseline, dtype = 'float'), np.array(perc + baseline, dtype = 'float'), color = colors[k], alpha = 0.5)
        baseline = baseline + perc
    axes1[1].set_title('DPC data')
    axes1[1].tick_params(axis='both', which='major', labelsize=7)
    axes1[1].tick_params(axis='both', which='minor', labelsize=7)
    axes1[1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    fig.legend(codes)
    fig.suptitle('Percentage of Infected', fontsize = 16)
    plt.savefig(ResultsFilePath + 'img/' +'perc_inf.png')
    plt.show()

    figg, axess1 = plt.subplots(1,2, figsize = (11,4))
    baseline = 0.0 * np.ones(len(date))
    for k in range(5):
        if k == 4:
            perc = (np.array(medians_age[k].Deceduti)+np.array(medians_age[k+1].Deceduti)) / medians_dec
        else:
            perc = np.array(medians_age[k].Deceduti) / medians_dec
        axess1[0].plot(datestr,perc + baseline, color = colors[k])
        axess1[0].fill_between(datestr, baseline, perc + baseline, color = colors[k], alpha = 0.5)
        baseline = baseline + perc
    axess1[0].set_title('Calibrated model')
    axess1[0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    axess1[0].tick_params(axis='both', which='major', labelsize=7)
    axess1[0].tick_params(axis='both', which='minor', labelsize=7)
    baseline = 0.0 * np.ones(len(date))
    for k in range(5):
        if k ==4:
            perc = (np.array(eData[eData.Age == ages[k]].Deceased)+np.array(eData[eData.Age == ages[k+1]].Deceased)) / np.array(eData.groupby(by = 'data').Deceased.sum())
        else:
            perc = np.array(eData[eData.Age == ages[k]].Deceased) / np.array(eData.groupby(by = 'data').Deceased.sum())
        axess1[1].plot(datestr,perc+ baseline, color = colors[k])
        axess1[1].fill_between(date, np.array(baseline, dtype = 'float'), np.array(perc + baseline, dtype = 'float'), color = colors[k], alpha = 0.5)
        baseline = baseline + perc
    axess1[1].set_title('DPC data')
    axess1[1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    axess1[1].tick_params(axis='both', which='major', labelsize=7)
    axess1[1].tick_params(axis='both', which='minor', labelsize=7)
    figg.legend(codes)
    figg.suptitle('Percentage of Deceased', fontsize = 16)
    plt.savefig(ResultsFilePath + 'img/' +'perc_dec.png')
    plt.show()
    
    
    fig, axes = plt.subplots(1,1, figsize = (8,7))
    Deceduti_tot = eData.groupby(by = 'data').Deceased.sum()
    
    axes.plot(datestr, Deceduti_tot, '*k')
    axes.plot(datestr, medians_dec, 'r')
    axes.legend(['DPC data', 'MCMC median'])
    axes.set_title('Total Deceased')
    axes.grid()
    axes.set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    axes.fill_between(date, fquant_dec, lquant_dec, alpha = 0.1, color = 'r')
    plt.savefig(ResultsFilePath + 'img/' +'dec_tot.png')
    plt.show()

    f = plt.figure(figsize = (30,12))
    ax = [plt.subplot(231+x) for x in range(5)]
    ax[0].plot(datestr, eData[eData.Age == '0-19'].Infected, '--k')
    ax[0].plot(eData[eData.Age == '0-19'].data, medians_age[0].Positivi, 'b')
    ax[0].fill_between(eData[eData.Age == '0-19'].data, fquant_age[0].Positivi, lquant_age[0].Positivi,\
            alpha = 0.1, color = 'b')
    ax[0].plot(datestr, simdf[simdf.Age == '0-19'].Infected, 'green')
    ax[0].set_title('0-19', fontsize = 8)
    ax[0].grid()
    ax[0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    date = eData[eData.Age == '0-19'].data

    ax[1].plot(date, eData[eData.Age == '20-39'].Infected, '--k')
    ax[1].plot(date, medians_age[1].Positivi, 'b')
    ax[1].fill_between(date, fquant_age[1].Positivi, lquant_age[1].Positivi,\
            alpha = 0.1, color = 'b')
    ax[1].plot(date, simdf[simdf.Age == '20-39'].Infected, 'green')
    ax[1].set_title('20-39', fontsize = 9)
    ax[1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    ax[1].grid()

    ax[2].plot(date, eData[eData.Age == '40-59'].Infected, '--k')
    ax[2].plot(date, medians_age[2].Positivi, 'b')
    ax[2].fill_between(date, fquant_age[2].Positivi, lquant_age[2].Positivi,\
            alpha = 0.1, color = 'b')
    ax[2].plot(date, simdf[simdf.Age == '40-59'].Infected, 'green')
    ax[2].set_title('40-59', fontsize = 9)
    ax[2].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    ax[2].grid()

    ax[3].plot(date, eData[eData.Age == '60-79'].Infected, '--k')
    ax[3].plot(date, medians_age[3].Positivi, 'b')
    ax[3].fill_between(date, fquant_age[3].Positivi, lquant_age[3].Positivi,\
            alpha = 0.1, color = 'b')
    ax[3].plot(date, simdf[simdf.Age == '60-79'].Infected, 'green')
    ax[3].set_title('60-79', fontsize = 9)
    ax[3].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    ax[3].grid()

    ax[4].plot(date, np.array(eData[eData.Age == '80-89'].Infected) + np.array(eData[eData.Age == '90+'].Infected), '--k')
    ax[4].plot(date, np.array(medians_age[4].Positivi) + np.array(medians_age[5].Positivi), 'b')
    ax[4].fill_between(date, np.array(fquant_age[4].Positivi) + np.array(fquant_age[5].Positivi), np.array(lquant_age[4].Positivi) + np.array(lquant_age[5].Positivi),\
            alpha = 0.1, color = 'b')
    ax[4].plot(date, np.array(simdf[simdf.Age == '80-89'].Infected) + np.array(simdf[simdf.Age == '90+'].Infected), 'green')
    ax[4].set_title('80+', fontsize = 9)
    ax[4].grid()
    f.legend(['DPC data', 'MCMC median', 'LS approx'])

    for i,x in enumerate(ax):
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    
    fig = plt.gcf()
    fig.set_size_inches((19.2,18.2), forward=False)
    plt.show()
    fig.savefig(ResultsFilePath + '/img/infected.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/infected.ptfig', 'wb'))



    f1 = plt.figure(figsize = (30,12))
    ax = [plt.subplot(231+x) for x in range(5)]
    ax[0].plot(datestr, eData[eData.Age == '0-19'].Deceased, '--k')
    ax[0].plot(eData[eData.Age == '0-19'].data, medians_age[0].Deceduti, 'r')
    ax[0].fill_between(eData[eData.Age == '0-19'].data, fquant_age[0].Deceduti, lquant_age[0].Deceduti,\
            alpha = 0.1, color = 'r')
    ax[0].plot(datestr, simdf[simdf.Age == '0-19'].Deceased, 'orange')
    ax[0].set_title('0-19', fontsize = 8)
    ax[0].grid()
    ax[0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    date = eData[eData.Age == '0-19'].data

    ax[1].plot(date, eData[eData.Age == '20-39'].Deceased, '--k')
    ax[1].plot(date, medians_age[1].Deceduti, 'r')
    ax[1].fill_between(date, fquant_age[1].Deceduti, lquant_age[1].Deceduti,\
            alpha = 0.1, color = 'r')
    ax[1].plot(date, simdf[simdf.Age == '20-39'].Deceased, 'orange')
    ax[1].set_title('20-39', fontsize = 9)
    ax[1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    ax[1].grid()

    ax[2].plot(date, eData[eData.Age == '40-59'].Deceased, '--k')
    ax[2].plot(date, medians_age[2].Deceduti, 'r')
    ax[2].fill_between(date, fquant_age[2].Deceduti, lquant_age[2].Deceduti,\
            alpha = 0.1, color = 'r')
    ax[2].plot(date, simdf[simdf.Age == '40-59'].Deceased, 'orange')
    ax[2].set_title('40-59', fontsize = 9)
    ax[2].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    ax[2].grid()

    ax[3].plot(date, eData[eData.Age == '60-79'].Deceased, '--k')
    ax[3].plot(date, medians_age[3].Deceduti, 'r')
    ax[3].fill_between(date, fquant_age[3].Deceduti, lquant_age[3].Deceduti,\
            alpha = 0.1, color = 'r')
    ax[3].plot(date, simdf[simdf.Age == '60-79'].Deceased, 'orange')
    ax[3].set_title('60-79', fontsize = 9)
    ax[3].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140))) 
    ax[3].grid()

    ax[4].plot(date, np.array(eData[eData.Age == '80-89'].Deceased) + np.array(eData[eData.Age == '90+'].Deceased), '--k')
    ax[4].plot(date, np.array(medians_age[4].Deceduti) + np.array(medians_age[5].Deceduti), 'r')
    ax[4].fill_between(date, np.array(fquant_age[4].Deceduti) + np.array(fquant_age[5].Deceduti), np.array(lquant_age[4].Deceduti) + np.array(lquant_age[5].Deceduti),\
            alpha = 0.1, color = 'r')
    ax[4].plot(date, np.array(simdf[simdf.Age == '80-89'].Deceased) + np.array(simdf[simdf.Age == '90+'].Deceased), 'orange')
    ax[4].set_title('80+', fontsize = 9)
    ax[4].grid()
    f1.legend(['DPC data', 'MCMC median', 'LS approx'])

    for i,x in enumerate(ax):
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    fig1 = plt.gcf()
    fig1.set_size_inches((19.2,18.2), forward=False)
    plt.show()
    fig1.savefig(ResultsFilePath + '/img/deceased.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/deceased.ptfig', 'wb'))

    f = plt.figure(2)
    ax = [plt.subplot(231+x) for x in range(6)]
    ax[0].plot(date, eData[eData.Age == '0-19'].Deceased, '--k')
    ax[0].plot(date, medians_age[0].Deceduti, '--b')
    ax[0].fill_between(date, fquant_age[0].Deceduti, lquant_age[0].Deceduti,\
            alpha = 0.2, color = 'b')
    ax[0].plot(date, simdf[simdf.Age == '0-19'].Deceased, '--r')
    ax[0].set_title('Deceased 0-19', fontsize = 9)
    ax[0].grid()

    ax[1].plot(date, eData[eData.Age == '20-39'].Deceased, '--k')
    ax[1].plot(date, medians_age[1].Deceduti, '--b')
    ax[1].fill_between(date, fquant_age[1].Deceduti, lquant_age[1].Deceduti,\
            alpha = 0.2, color = 'b')
    ax[1].plot(date, simdf[simdf.Age == '20-39'].Deceased, '--r')
    ax[1].set_title('Deceased 20-39', fontsize = 9)
    ax[1].grid()

    ax[2].plot(date, eData[eData.Age == '40-59'].Deceased, '--k')
    ax[2].plot(date, medians_age[2].Deceduti, '--b')
    ax[2].fill_between(date, fquant_age[2].Deceduti, lquant_age[2].Deceduti,\
            alpha = 0.2, color = 'b')
    ax[2].plot(date, simdf[simdf.Age == '40-59'].Deceased, '--r')
    ax[2].set_title('Deceased 40-59', fontsize = 9)
    ax[2].grid()

    ax[3].plot(date, eData[eData.Age == '60-79'].Deceased, '--k')
    ax[3].plot(date, medians_age[3].Deceduti, '--b')
    ax[3].fill_between(date, fquant_age[3].Deceduti, lquant_age[3].Deceduti,\
            alpha = 0.2, color = 'b')
    ax[3].plot(date, simdf[simdf.Age == '60-79'].Deceased, '--r')
    ax[3].set_title('Deceased 60-79', fontsize = 9)
    ax[3].grid()

    ax[4].plot(date, eData[eData.Age == '80-89'].Deceased, '--k')
    ax[4].plot(date, medians_age[4].Deceduti, '--b')
    ax[4].fill_between(date, fquant_age[4].Deceduti, lquant_age[4].Deceduti,\
            alpha = 0.2, color = 'b')
    ax[4].plot(date, simdf[simdf.Age == '80-89'].Deceased, '--r')
    ax[4].set_title('Deceased 80-89', fontsize = 9)
    ax[4].grid()

    ax[5].plot(date, eData[eData.Age == '90+'].Deceased, '--k')
    ax[5].plot(date, medians_age[5].Deceduti, '--b')
    ax[5].fill_between(date, fquant_age[5].Deceduti, lquant_age[5].Deceduti,\
            alpha = 0.2, color = 'b')
    ax[5].plot(date, simdf[simdf.Age == '90+'].Deceased, '--r')
    ax[5].set_title('Deceased 90+', fontsize = 9)
    ax[5].grid()

    for i,x in enumerate(ax):
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    fig = plt.gcf()
    fig.set_size_inches((19.2,18.2), forward=False)
    plt.show()
    fig.savefig(ResultsFilePath + '/img/deceased.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/deceased.ptfig', 'wb'))
    
    f = plt.figure(3)
    ax = [plt.subplot(231+x) for x in range(6)]
    ax[0].plot(date, medians_age[0]['Vaccinati Prima Dose'], '--b')
    ax[0].fill_between(date, fquant_age[0]['Vaccinati Prima Dose'], lquant_age[0]['Vaccinati Prima Dose'],\
            alpha = 0.2, color = 'b')
    ax[0].plot(date, simdf[simdf.Age == '0-19'].VaccinatedFirst, '--r')
    ax[0].set_title('Vaccinated First dose 0-19', fontsize = 9)
    ax[0].grid()

    ax[1].plot(date, medians_age[1]['Vaccinati Prima Dose'], '--b')
    ax[1].fill_between(date, fquant_age[1]['Vaccinati Prima Dose'], lquant_age[1]['Vaccinati Prima Dose'],\
            alpha = 0.2, color = 'b')
    ax[1].plot(date, simdf[simdf.Age == '20-39'].VaccinatedFirst, '--r')
    ax[1].set_title('Vaccinated First 20-39', fontsize = 9)
    ax[1].grid()

    ax[2].plot(date, medians_age[2]['Vaccinati Prima Dose'], '--b')
    ax[2].fill_between(date, fquant_age[2]['Vaccinati Prima Dose'], lquant_age[2]['Vaccinati Prima Dose'],\
            alpha = 0.2, color = 'b')
    ax[2].plot(date, simdf[simdf.Age == '40-59'].VaccinatedFirst, '--r')
    ax[2].set_title('Vaccinated First 40-59', fontsize = 9)
    ax[2].grid()

    ax[3].plot(date, medians_age[3]['Vaccinati Prima Dose'], '--b')
    ax[3].fill_between(date, fquant_age[3]['Vaccinati Prima Dose'], lquant_age[3]['Vaccinati Prima Dose'],\
            alpha = 0.2, color = 'b')
    ax[3].plot(date, simdf[simdf.Age == '60-79'].VaccinatedFirst, '--r')
    ax[3].set_title('Vaccinated First 60-79', fontsize = 9)
    ax[3].grid()

    ax[4].plot(date, medians_age[4]['Vaccinati Prima Dose'], '--b')
    ax[4].fill_between(date, fquant_age[4]['Vaccinati Prima Dose'], lquant_age[4]['Vaccinati Prima Dose'],\
            alpha = 0.2, color = 'b')
    ax[4].plot(date, simdf[simdf.Age == '80-89'].VaccinatedFirst, '--r')
    ax[4].set_title('Vaccinated First 80-89', fontsize = 9)
    ax[4].grid()

    ax[5].plot(date, medians_age[5]['Vaccinati Prima Dose'], '--b')
    ax[5].fill_between(date, fquant_age[5]['Vaccinati Prima Dose'], lquant_age[5]['Vaccinati Prima Dose'],\
            alpha = 0.2, color = 'b')
    ax[5].plot(date, simdf[simdf.Age == '90+'].VaccinatedFirst, '--r')
    ax[5].set_title('Vaccinated First 90+', fontsize = 9)
    ax[5].grid()

    for i,x in enumerate(ax):
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    fig = plt.gcf()
    fig.set_size_inches((19.2,18.2), forward=False)
    plt.show()
    fig.savefig(ResultsFilePath + '/img/vaccinatedfirst.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/vaccinatedfirst.ptfig', 'wb'))
    

    f = plt.figure(4)
    ax = [plt.subplot(231+x) for x in range(6)]
    ax[0].plot(date, medians_age[0]['Vaccinati Ciclo Completo'], '--b')
    ax[0].fill_between(date, fquant_age[0]['Vaccinati Ciclo Completo'], lquant_age[0]['Vaccinati Ciclo Completo'],\
            alpha = 0.2, color = 'b')
    ax[0].plot(date, simdf[simdf.Age == '0-19'].VaccinatedSecond, '--r')
    ax[0].set_title('Vaccinated Second 0-19', fontsize = 9)
    ax[0].grid()

    ax[1].plot(date, medians_age[1]['Vaccinati Ciclo Completo'], '--b')
    ax[1].fill_between(date, fquant_age[1]['Vaccinati Ciclo Completo'], lquant_age[1]['Vaccinati Ciclo Completo'],\
            alpha = 0.2, color = 'b')
    ax[1].plot(date, simdf[simdf.Age == '20-39'].VaccinatedSecond, '--r')
    ax[1].set_title('Vaccinated Second 20-39', fontsize = 9)
    ax[1].grid()

    ax[2].plot(date, medians_age[2]['Vaccinati Ciclo Completo'], '--b')
    ax[2].fill_between(date, fquant_age[2]['Vaccinati Ciclo Completo'], lquant_age[2]['Vaccinati Ciclo Completo'],\
            alpha = 0.2, color = 'b')
    ax[2].plot(date, simdf[simdf.Age == '40-59'].VaccinatedSecond, '--r')
    ax[2].set_title('Vaccinated Second 40-59', fontsize = 9)
    ax[2].grid()

    ax[3].plot(date, medians_age[3]['Vaccinati Ciclo Completo'], '--b')
    ax[3].fill_between(date, fquant_age[3]['Vaccinati Ciclo Completo'], lquant_age[3]['Vaccinati Ciclo Completo'],\
            alpha = 0.2, color = 'b')
    ax[3].plot(date, simdf[simdf.Age == '60-79'].VaccinatedSecond, '--r')
    ax[3].set_title('Vaccinated Second 60-79', fontsize = 9)
    ax[3].grid()

    ax[4].plot(date, medians_age[4]['Vaccinati Ciclo Completo'], '--b')
    ax[4].fill_between(date, fquant_age[4]['Vaccinati Ciclo Completo'], lquant_age[4]['Vaccinati Ciclo Completo'],\
            alpha = 0.2, color = 'b')
    ax[4].plot(date, simdf[simdf.Age == '80-89'].VaccinatedSecond, '--r')
    ax[4].set_title('Vaccinated Second 80-89', fontsize = 9)
    ax[4].grid()

    ax[5].plot(date, medians_age[5]['Vaccinati Ciclo Completo'], '--b')
    ax[5].fill_between(date, fquant_age[5]['Vaccinati Ciclo Completo'], lquant_age[5]['Vaccinati Ciclo Completo'],\
            alpha = 0.2, color = 'b')
    ax[5].plot(date, simdf[simdf.Age == '90+'].VaccinatedSecond, '--r')
    ax[5].set_title('Vaccinated Second 90+', fontsize = 9)
    ax[5].grid()

    for i,x in enumerate(ax):
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    fig = plt.gcf()
    fig.set_size_inches((19.2,18.2), forward=False)
    plt.show()
    fig.savefig(ResultsFilePath + '/img/vaccinatedsecond.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/vaccinatedsecond.ptfig', 'wb'))
    
    f = plt.figure(5)
    ax = [plt.subplot(231+x) for x in range(6)]
    ax[0].plot(date, medians_age[0].Suscettibili, '--b')
    ax[0].fill_between(date, fquant_age[0].Suscettibili, lquant_age[0].Suscettibili,\
            alpha = 0.2, color = 'b')
    ax[0].plot(date, simdf[simdf.Age == '0-19'].Suscept, '--r')
    ax[0].set_title('Susceptibles 0-19', fontsize = 9)
    ax[0].grid()

    ax[1].plot(date, medians_age[1].Suscettibili, '--b')
    ax[1].fill_between(date, fquant_age[1].Suscettibili, lquant_age[1].Suscettibili,\
            alpha = 0.2, color = 'b')
    ax[1].plot(date, simdf[simdf.Age == '20-39'].Suscept, '--r')
    ax[1].set_title('Susceptibles 20-39', fontsize = 9)
    ax[1].grid()

    ax[2].plot(date, medians_age[2].Suscettibili, '--b')
    ax[2].fill_between(date, fquant_age[2].Suscettibili, lquant_age[2].Suscettibili,\
            alpha = 0.2, color = 'b')
    ax[2].plot(date, simdf[simdf.Age == '40-59'].Suscept, '--r')
    ax[2].set_title('Susceptibles 40-59', fontsize = 9)
    ax[2].grid()

    ax[3].plot(date, medians_age[3].Suscettibili, '--b')
    ax[3].fill_between(date, fquant_age[3].Suscettibili, lquant_age[3].Suscettibili,\
            alpha = 0.2, color = 'b')
    ax[3].plot(date, simdf[simdf.Age == '60-79'].Suscept, '--r')
    ax[3].set_title('Susceptibles 60-79', fontsize = 9)
    ax[3].grid()

    ax[4].plot(date, medians_age[4].Suscettibili, '--b')
    ax[4].fill_between(date, fquant_age[4].Suscettibili, lquant_age[4].Suscettibili,\
            alpha = 0.2, color = 'b')
    ax[4].plot(date, simdf[simdf.Age == '80-89'].Suscept, '--r')
    ax[4].set_title('Susceptibles 80-89', fontsize = 9)
    ax[4].grid()

    ax[5].plot(date, medians_age[5].Suscettibili, '--b')
    ax[5].fill_between(date, fquant_age[5].Suscettibili, lquant_age[5].Suscettibili,\
            alpha = 0.2, color = 'b')
    ax[5].plot(date, simdf[simdf.Age == '90+'].Suscept, '--r')
    ax[5].set_title('Susceptibles 90+', fontsize = 9)
    ax[5].grid()

    for i,x in enumerate(ax):
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    fig = plt.gcf()
    fig.set_size_inches((19.2,18.2), forward=False)
    plt.show()
    fig.savefig(ResultsFilePath + '/img/susc.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/susc.ptfig', 'wb'))
    
    f = plt.figure(6)
    ax = [plt.subplot(231+x) for x in range(6)]
    ax[0].plot(date, medians_age[0].Guariti, '--b')
    ax[0].fill_between(date, fquant_age[0].Guariti, lquant_age[0].Guariti,\
            alpha = 0.2, color = 'b')
    ax[0].plot(date, simdf[simdf.Age == '0-19'].Recovered, '--r')
    ax[0].set_title('Recovered 0-19', fontsize = 9)
    ax[0].grid()

    ax[1].plot(date, medians_age[1].Guariti, '--b')
    ax[1].fill_between(date, fquant_age[1].Guariti, lquant_age[1].Guariti,\
            alpha = 0.2, color = 'b')
    ax[1].plot(date, simdf[simdf.Age == '20-39'].Recovered, '--r')
    ax[1].set_title('Recovered 20-39', fontsize = 9)
    ax[1].grid()

    ax[2].plot(date, medians_age[2].Guariti, '--b')
    ax[2].fill_between(date, fquant_age[2].Guariti, lquant_age[2].Guariti,\
            alpha = 0.2, color = 'b')
    ax[2].plot(date, simdf[simdf.Age == '40-59'].Recovered, '--r')
    ax[2].set_title('Recovered 40-59', fontsize = 9)
    ax[2].grid()

    ax[3].plot(date, medians_age[3].Guariti, '--b')
    ax[3].fill_between(date, fquant_age[3].Guariti, lquant_age[3].Guariti,\
            alpha = 0.2, color = 'b')
    ax[3].plot(date, simdf[simdf.Age == '60-79'].Recovered, '--r')
    ax[3].set_title('Recovered 60-79', fontsize = 9)
    ax[3].grid()

    ax[4].plot(date, medians_age[4].Guariti, '--b')
    ax[4].fill_between(date, fquant_age[4].Guariti, lquant_age[4].Guariti,\
            alpha = 0.2, color = 'b')
    ax[4].plot(date, simdf[simdf.Age == '80-89'].Recovered, '--r')
    ax[4].set_title('Recovered 80-89', fontsize = 9)
    ax[4].grid()

    ax[5].plot(date, medians_age[5].Guariti, '--b')
    ax[5].fill_between(date, fquant_age[5].Guariti, lquant_age[5].Guariti,\
            alpha = 0.2, color = 'b')
    ax[5].plot(date, simdf[simdf.Age == '90+'].Recovered, '--r')
    ax[5].set_title('Recovered 90+', fontsize = 9)
    ax[5].grid()

    for i,x in enumerate(ax):
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    fig = plt.gcf()
    fig.set_size_inches((15.2,10.2), forward=False)
    plt.show()
    fig.savefig(ResultsFilePath + '/img/rec.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/rec.ptfig', 'wb'))
    
    f = plt.figure(7)
    ax = [plt.subplot(231+x) for x in range(6)]
    date = eData[eData.Age == '0-19'].data
    ax[0].plot(date, param_matrix[:,0], '--b')
    ax[0].set_title('T-Rate 0-19', fontsize = 9)
    ax[0].grid()

    ax[1].plot(date, param_matrix[:,1], '--b')
    ax[1].set_title('T-Rate 20-39', fontsize = 9)
    ax[1].grid()

    ax[2].plot(date, param_matrix[:,2], '--b')
    ax[2].set_title('T-Rate 40-59', fontsize = 9)
    ax[2].grid()

    ax[3].plot(date, param_matrix[:,3], '--b')
    ax[3].set_title('T-Rate 60-79', fontsize = 9)
    ax[3].grid()

    ax[4].plot(date, param_matrix[:,4], '--b')
    ax[4].set_title('T-Rate 80-89', fontsize = 9)
    ax[4].grid()

    ax[5].plot(date, param_matrix[:,5], '--b')
    ax[5].set_title('T-Rate 90+', fontsize = 9)
    ax[5].grid()

    for i,x in enumerate(ax):
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='small')
        handles, labels = x.get_legend_handles_labels()
    fig = plt.gcf()
    fig.set_size_inches((19.2,18.2), forward=False)
    plt.show()
    fig.savefig(ResultsFilePath + '/img/beta.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/beta.ptfig', 'wb'))
