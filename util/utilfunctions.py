import os.path
import zipfile
import pandas as pd
import numpy as np
import datetime
from epi.convert import converter
from scipy import interpolate

def zipdir(path, ziph):
    # ziph is zipfile handle
    exclude = ['Tests','__pycache__','venv','.git','.gitignore','forecast_hub','forecast_hub_images']
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

def correct_isolated_model(df, comp, date, DT, C=None, total=None, healed=None):
    if not C:
        C = df.loc[date,comp]-df.loc[date-pd.Timedelta(1,'day'),comp]
    Itot = df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp].sum()
    Cn =df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp]/ Itot * C
    df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp] += Cn.cumsum()
    if total:
        df.loc[date - pd.Timedelta(DT, 'days'):date - pd.Timedelta(1, 'day'), total] += Cn.cumsum()
    if healed:
        df.loc[date - pd.Timedelta(DT, 'days'):date - pd.Timedelta(1, 'day'), healed] -= Cn.cumsum()
    return df

def correct_data(eData, country):
    corrections = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/corrections.csv',keep_default_na=False,na_values=['NaN'])
    if country != 'Italia':
        corrections = corrections[corrections['region']==country]
    corrections.replace({'NaN': None})
    comp_ita =  [
            'Positivi',
            'Nuovi positivi',
            'Individui in isolamento domiciliare',
            'Ricoverati (non in terapia intensiva)',
            'Ricoverati in terapia intensiva',
            'Nuovi ingressi in terapia intensiva',
            'Guariti',
            'Deceduti',
            'Deceduti giornalieri',
            'Tamponi eseguiti',
            'Nuovi tamponi eseguiti'
    ]
    old_comp = [
            'Totale positivi',
            'Nuovi positivi',
            'Isolamento domiciliare',
            'Ricoverati con sintomi',
            'Terapia intensiva',
            'Ingressi terapia intensiva',
            'Dimessi guariti',
            'Deceduti',
            'Nuovi deceduti',
            'Tamponi',
            'Nuovi tamponi',
    ]
    comp_to_old = dict(zip(comp_ita, old_comp))

    for _, row in corrections.iterrows():
        eData = correct_isolated_model(eData.set_index('data'), comp_to_old[row['comp']].replace(' ', '_').lower(),
                                      pd.to_datetime(row['date']), int(row['DT']), int(row['C']),
                                      comp_to_old[row['Positivi']].replace(' ', '_').lower(),
                                      comp_to_old[row['Guariti']].replace(' ', '_').lower()).reset_index().rename(
            columns={'index': 'data'})
    return eData

def estimate_IFR(country, Pop):
    day_ISS_data = pd.to_datetime('2020-12-08')
    IFR_age = np.array([0.005, 0.015, 0.035, 0.08, 0.2, 0.49, 1.205, 2.96, 7.26, 17.37])/100
    m1=[1, 1, 1, 1, 0.848, 0.848, 0.697, 0.697, 0.545, 0.545]
    m2=[1, 1, 1, 1, 0.553, 0.553, 0.787, 0.787, 0.411, 0.411]
    vaccini_by_age = pd.read_csv("https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccini_regioni/"+country.replace(' ','%20')+"_age.csv")
    vaccini_by_age['data'] = pd.to_datetime(vaccini_by_age.data)
    vaccini_by_age = [v.set_index('data').reindex(pd.date_range(day_ISS_data,max(v.data))).fillna(0) for i,v in vaccini_by_age.groupby('eta')]
    pop_age = np.array([8.4, 9.6, 10.3, 11.7, 15.3, 15.4, 12.2, 9.9, 5.9, 1.3])/100 * Pop
    age = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    infected_age = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico_all_ages.csv')
    infected_age['Data'] = pd.to_datetime(infected_age.Data)

    infected_age = [v.set_index('Data') for i,v in infected_age.groupby('Età')]
    IFR_t = pd.Series(np.zeros(len(infected_age[0].index)),index=infected_age[0].index)
    for i,IFR in enumerate(IFR_age):
        if i==0:
            IFR_t += infected_age[0].Infected * IFR_age[0]
        else:
            IFR_t += (infected_age[i].Infected * IFR_age[i] * (pop_age[i]-vaccini_by_age[i-1].prima_dose.cumsum()) +\
                     infected_age[i].Infected * IFR_age[i] * m1[i] * (vaccini_by_age[i-1].prima_dose.cumsum() - vaccini_by_age[i-1].seconda_dose.cumsum()) +\
                     infected_age[i].Infected * IFR_age[i] * m2[i] *  vaccini_by_age[i-1].seconda_dose.cumsum()) / pop_age[i]
    IFR_t.loc[pd.to_datetime('2020-02-24')] = 0.014
    IFR_t = IFR_t.reindex(index=pd.date_range(min(IFR_t.index), max(IFR_t.index))).ffill()
    return IFR_t

def estimate_IFR_age(country, Pop):
    day_ISS_data = pd.to_datetime('2020-12-08')
    Pop_all_ages = np.array([0.081, 0.096, 0.101, 0.113, 0.127, 0.159, 0.125, 0.101, 0.062, 0.013, 0.00029])
    IFR_age = np.array([0.005* Pop_all_ages[0] + 0.015*Pop_all_ages[1], 0.035*Pop_all_ages[2] + 0.08*Pop_all_ages[3],\
            0.2*Pop_all_ages[4] + 0.49*Pop_all_ages[5], 1.205*Pop_all_ages[6]+ 2.96*Pop_all_ages[7], 7.26, 17.37])/100
    m1=[1,  1,  0.848,  0.697, 0.545, 0.545]
    m2=[1,  1,  0.553,  0.787, 0.411, 0.411]
    vaccini_by_age = pd.read_csv("https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccini_regioni/"+country.replace(' ','%20')+"_age.csv")
    vaccini_by_age['data'] = pd.to_datetime(vaccini_by_age.data)
    vaccini_by_age = [v.set_index('data').reindex(pd.date_range(day_ISS_data,max(v.data))).fillna(0) for i,v in vaccini_by_age.groupby('eta')]
    pop_age = np.array([18.0, 22.0, 30.7, 22.1, 5.9, 1.3])/100 * Pop
    age = [10, 30, 50, 70, 85, 95]
    infected_age = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico_all_ages.csv')
    infected_age['Data'] = pd.to_datetime(infected_age.Data)

    infected_age = [v.set_index('Data') for i,v in infected_age.groupby('Età')]
    IFR_t = []
    IFR_t_i = pd.Series(np.zeros(len(infected_age[0].index)),index=infected_age[0].index)
    for i,IFR in enumerate(IFR_age):
        if i==0:
            IFR_t_i = infected_age[0].Infected * IFR_age[0]
        else:
            IFR_t_i = (infected_age[i].Infected * IFR_age[i] * (pop_age[i]-vaccini_by_age[i-1].prima_dose.cumsum()) +\
                     infected_age[i].Infected * IFR_age[i] * m1[i] * (vaccini_by_age[i-1].prima_dose.cumsum() - vaccini_by_age[i-1].seconda_dose.cumsum()) +\
                     infected_age[i].Infected * IFR_age[i] * m2[i] *  vaccini_by_age[i-1].seconda_dose.cumsum()) / pop_age[i]
        IFR_t_i.loc[pd.to_datetime('2020-02-24')] = 0.014
        IFR_t_i = IFR_t_i.reindex(index=pd.date_range(min(IFR_t_i.index), max(IFR_t_i.index))).ffill()
        IFR_t.append(IFR_t_i)
    return IFR_age


def estimate_CFR(eData):
    window = 7
    deltaD = eData['deceduti'].diff(periods=2*window).shift(-window-13)
    deltaR = eData['dimessi_guariti'].diff(periods=2*window).shift(-window-13)
    CFR_t = deltaD/(deltaD+deltaR)
    return CFR_t

def sum_vec(x, x_incr):
    return x + x_incr

def estimate_CFR_age(country, day_init, day_end, DPC_start, DPC_end, Nc, Ns):
    if country == 'Regions': 
        eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    else:
        eData = pd.read_csv("https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/regioni/"+country.replace(' ','%20')+".csv")
    
    eData['data'] = [x[:10] for x in eData.data]
    eData['data'] = pd.to_datetime(eData.data)
    
    eData = correct_data(eData, country)
    
    eData = eData[(eData["data"]>=day_init.isoformat()) & (eData["data"]<=(day_end+ datetime.timedelta(20)).isoformat())]
    eData = eData.reset_index(drop=True)
    eData = converter('SUIHTER', eData, country, Nc)
    eData = eData.reset_index(drop=True)

    perc = pd.read_csv(f'~/dpc-covid-data/SUIHTER/stato_clinico.csv')
    #perc = pd.read_csv(f'https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico_{model}.csv')
    perc = perc[(perc['Data']>=DPC_start) & (perc['Data']<=(day_end+ datetime.timedelta(20)).strftime('%Y-%m-%d'))]
    eData = pd.DataFrame(np.repeat(eData.values,Ns,axis=0),columns=eData.columns)
    eData[perc.columns[2:]] = eData[perc.columns[2:]].mul(perc[perc.columns[2:]].values)
    eData['Age'] = perc['Età'].values
    eData.sort_values(by=['Age','time'])
    eData = eData.iloc[:, 4:]
    window = 7
    CFR = []
    for age in ['0-19', '20-39', '40-59', '60-79', '80-89', '90+']:
        data = eData[eData['Age'] == age]
        data = data.iloc[:,:8]
        if age == '0-19':
            data['Extinct'] = data['Daily_extinct'].cumsum(axis = 0) + data.loc[0, 'Extinct']
            #print(np.array(data.Extinct).shape)
            #print(np.arange((day_end-day_init).days+window + 13 +1).shape)
            days = np.arange((day_end - day_init).days +1 +13 +window)
            extinct = np.array(data.Extinct).astype(float)
            p1 = np.polyfit(days , extinct,2)
            deltaD = p1[0] * days **2 + p1[1] * days + p1[2]
            #deltaD = p1[0] * days *3 + p1[1] * days**2 + p1[2]*days + p1[3]
            #print(data.loc[:,'Extinct'])
            #print(deltaD)
            #print('Rec', data.Recovered)
        else: 
            deltaD = data['Extinct'].diff(periods = 2*window).shift(-window-13).to_numpy()
        deltaR = data.Recovered.diff(periods =  2*window).shift(-window-13).to_numpy()
        CFR_t = deltaD / (deltaD + deltaR)
        print(CFR_t.shape)
        CFR_t = CFR_t[:-window -13]
        CFR.append(CFR_t)
    return CFR
