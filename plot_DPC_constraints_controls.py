import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pltlab
import matplotlib as mpl
import matplotlib.ticker as mticker
import datetime
import numpy as np

#f_name = 'Tests/SIRDVW_age_OC_2022-07-05_ITA_Cost1Ages1/'#Inf
f_name_IC = 'Tests/SIRDVW_age_OC_2022-09-01_constraintsITA_IC_homo_Cost1/'
#f_name = 'Tests/SIRDVW_age_OC_2022-07-05_ITA_Cost2Ages1/'#Dec 
#f_name = 'Tests/SIRDVW_age_OC_2022-07-05_ITA_Cost3Ages1/'#Hosp 
f_name_base_IC = 'Tests/SIRDVW_age_OC_reference_2022-09-08_HomoIC/'
f_name = 'Tests/SIRDVW_age_OC_2022-07-05_ITA_Cost1Ages1/'#Inf
f_name_base = 'Tests/SIRDVW_age_OC_reference_2022-09-08_stdIC/'

#US1,... simulazione IC non reali
#US1_r,... dosi iniziali per simulazione IC non reali
#US1_old,... Optimal policy partendo da dosi homo
#US1_r_old,... IC caso non reale
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
#plt.rc('legend',fontsize='xx-large')


US1 = pd.read_hdf(f_name + 'US1.h5').to_numpy()
US2 = pd.read_hdf(f_name + 'US2.h5').to_numpy()
UR = pd.read_hdf(f_name + 'UR.h5').to_numpy()
plt.rc('legend',fontsize='xx-large')
day_init = pd.to_datetime('2021-02-12')
###################################################################day_init = pd.to_datetime('2021-01-01')
day_init_vaccines = day_init# - datetime.timedelta(14)
Tf_data = '2021-05-27'
vaccines = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccines_age.csv') 
vaccines['data'] = pd.to_datetime(vaccines.data) 
vaccines.set_index(['data', 'eta'],inplace=True) 
vaccines.fillna(0,inplace=True) 
#vaccines[['prima_dose','seconda_dose']]=0 
vaccines_init = vaccines[:day_init_vaccines-pd.Timedelta(1,'day')].sum() 
vaccines = vaccines.loc[day_init_vaccines:Tf_data] 
length = 105 # 147
I = np.zeros((length, int(length/7)))
for i in range(int(length/7)):
    I[i*7:(i+1)*7, i] = 1
#print(vaccines['prima_dose'])
US1_t = np.reshape(vaccines['prima_dose'].to_numpy(), (9, length), order = 'F')
US2_t = np.reshape(vaccines['seconda_dose'].to_numpy() + vaccines['mono_dose'].to_numpy(), (9, length), order = 'F')
UR_t = np.reshape(vaccines['pregressa_infezione'].to_numpy(), (9, length ), order = 'F')

US1_r = np.zeros((6, length))
US1_r[0] = US1_t[0]
US1_r[1] = np.sum(US1_t[1:3], axis = 0)
US1_r[2] = np.sum(US1_t[3:5], axis = 0)
US1_r[3] = np.sum(US1_t[5:7], axis = 0)
US1_r[4] = US1_t[7]
US1_r[5] = US1_t[8]
US1_r = US1_r @ I #controllare il dato finale
#self.US1 = np.zeros((6, ncolreduct))
US2_r = np.zeros((6, length))
US2_r[0] = US2_t[0]
US2_r[1] = np.sum(US2_t[1:3], axis = 0)
US2_r[2] = np.sum(US2_t[3:5], axis = 0)
US2_r[3] = np.sum(US2_t[5:7], axis = 0)
US2_r[4] = US2_t[7]
US2_r[5] = US2_t[8]
US2_r = US2_r @ I
#self.US2 = np.zeros((6, ncolreduct))
UR_r = np.zeros((6, length))
UR_r[0] = UR_t[0]
UR_r[1] = np.sum(UR_t[1:3], axis = 0)
UR_r[2] = np.sum(UR_t[3:5], axis = 0)
UR_r[3] = np.sum(UR_t[5:7], axis = 0)
UR_r[4] = UR_t[7]
UR_r[5] = UR_t[8]
UR_r = UR_r @ I
Ns = 5
ages_opt = [0,1,2,3,4]
delay_time = 3
ncolreduct = int(length/7)
###############################################################################RIMETTERE NCOLREDUCT AL POSTO LENGTH
US1_r = np.append(US1_r[:4,:], np.add.reduce(US1_r[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0)
US2_r = np.append(US2_r[:4,:], np.add.reduce(US2_r[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0)
US2_r[ages_opt,delay_time:] = US1_r[ages_opt,:-delay_time]
UR_r = np.append(UR_r[:4,:], np.add.reduce(UR_r[4:,:], axis = 0).reshape(1, ncolreduct), axis = 0)

US1_r_old = pd.read_hdf(f_name_base_IC + 'US1.h5').to_numpy()
US2_r_old = pd.read_hdf(f_name_base_IC + 'US2.h5').to_numpy()
UR_r_old  = pd.read_hdf(f_name_base_IC + 'UR.h5').to_numpy()

US1_old = pd.read_hdf(f_name_IC + 'US1.h5').to_numpy()
US2_old = pd.read_hdf(f_name_IC + 'US2.h5').to_numpy()
UR_old  = pd.read_hdf(f_name_IC + 'UR.h5').to_numpy()

Nc = 6 #Compartments 
codes = ['0-19', '20-39', '40-59', '60-79', '80+']

day_init_2 = pd.to_datetime('2021-01-01')
colors = plt.cm.cool(np.linspace(0, 1, 5))
date_str = [day_init + datetime.timedelta(k) for k in range(length)]

weeks = int(length/7)
baseline = 0.0 * np.ones(weeks-1)
ages = codes 
datestr = [day_init + datetime.timedelta(k) for k in range(length)]

date = np.arange(weeks)
US1_r_tot = np.sum(US1_r, axis = 0)
length2 = 98 
dates_list = [day_init + datetime.timedelta(x) for x in range(length2)]

weeks_list = [day_init + datetime.timedelta(7*x) for x in range(int(length2/7))]
colors = plt.cm.cool(np.linspace(0, 1, 5))

fig5, ax5 = plt.subplots(2,3, figsize=(30,10))
plt.xticks(fontsize = 12)
plt.yticks(fontsize=12)

ax5[0,0].plot(dates_list, np.repeat(US1[0,:-1],7), color = colors[1])
#ax5[0,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[0,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,0].plot(dates_list, np.repeat(US1_r[0,:-1], 7), '--', color = colors[1])
#ax5[0,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[0,:-1],7), color = 'orange', alpha = 0.08)

ax5[0,0].plot(dates_list, np.repeat(US1_old[0,:-1],7), color = colors[4])
#ax5[0,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[0,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,0].plot(dates_list, np.repeat(US1_r_old[0,:-1], 7), '--', color = colors[4])

ax5[0,0].set_title('0-19')
#ax5[0,0].legend(['Optimal policy', 'Initial policy'])
ax5[0,0].grid()
ax5[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(g))
ax5[0,0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[0,1].plot(dates_list, np.repeat(US1[1,:-1], 7), color = colors[1])
#ax5[0,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[1,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,1].plot(dates_list, np.repeat(US1_r[1,:-1], 7), '--', color = colors[1])
#ax5[0,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[1,:-1],7), color = 'orange', alpha = 0.08)

ax5[0,1].plot(dates_list, np.repeat(US1_old[1,:-1], 7), color = colors[4])
#ax5[0,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[1,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,1].plot(dates_list, np.repeat(US1_r_old[1,:-1], 7), '--', color = colors[4])

ax5[0,1].set_title('20-39')
#ax5[0,1].legend(['Optimal policy', 'Initial policy'])
ax5[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(g))
ax5[0,1].grid()
ax5[0,1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[0,2].plot(dates_list, np.repeat(US1[2,:-1], 7), color = colors[1])
#ax5[0,2].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[2,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,2].plot(dates_list, np.repeat(US1_r[2,:-1],7), '--', color = colors[1])
#ax5[0,2].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[2,:-1],7), color = 'orange', alpha = 0.08)

ax5[0,2].plot(dates_list, np.repeat(US1_old[2,:-1], 7), color = colors[4])
#ax5[0,2].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[2,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,2].plot(dates_list, np.repeat(US1_r_old[2,:-1],7), '--', color = colors[4])

ax5[0,2].yaxis.set_major_formatter(mticker.FuncFormatter(g))
ax5[0,2].set_title('40-59')
#ax5[0,2].legend(['Optimal policy', 'Initial policy'])
ax5[0,2].grid()
ax5[0,2].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,0].plot(dates_list, np.repeat(US1[3,:-1], 7), color = colors[1])
#ax5[1,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[3,:-1],7), color = colors[1], alpha = 0.08)
ax5[1,0].plot(dates_list, np.repeat(US1_r[3,:-1],7), '--', color = colors[1])
#ax5[1,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[3,:-1],7), color = 'orange', alpha = 0.08)

ax5[1,0].plot(dates_list, np.repeat(US1_old[3,:-1], 7), color = colors[4])
#ax5[1,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[3,:-1],7), color = colors[1], alpha = 0.08)
ax5[1,0].plot(dates_list, np.repeat(US1_r_old[3,:-1],7), '--', color = colors[4])
#ax5[1,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[3,:-1],7), color = 'orange', alpha = 0.08)

ax5[1,0].set_title('60-79')
#ax5[1,0].legend(['Optimal policy', 'Initial policy'])
ax5[1,0].yaxis.set_major_formatter(mticker.FuncFormatter(g))
ax5[1,0].grid()
ax5[1,0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,1].plot(dates_list, np.repeat(US1[4,:-1], 7), color = colors[1])
#ax5[1,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[4,:-1],7), color = colors[1], alpha = 0.08)
ax5[1,1].plot(dates_list, np.repeat(US1_r[4,:-1],7), '--', color = colors[1])
#ax5[1,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[4,:-1],7), color = 'orange', alpha = 0.08)

ax5[1,1].plot(dates_list, np.repeat(US1_old[4,:-1], 7), color = colors[4])
#ax5[1,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[4,:-1],7), color = colors[1], alpha = 0.08)
ax5[1,1].plot(dates_list, np.repeat(US1_r_old[4,:-1],7), '--', color = colors[4])

ax5[1,1].yaxis.set_major_formatter(mticker.FuncFormatter(g))
ax5[1,1].set_title('80+')
#ax5[1,1].legend(['Optimal policy', 'Initial policy'])
ax5[1,1].grid()
ax5[1,1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,2].remove()

fig5.legend(['Optimal policy - DPC IG', 'Initial policy - DPC IG', 'Optimal policy - Homogeneous IG', 'Initial policy - Homogeneous IG'], bbox_to_anchor=(0.75, 0.25), loc = 'center')
#plt.suptitle('US1')
plt.savefig(f_name_IC+'/img/US1_COMPARE.png')
plt.show()


plt.close('all')

#f_name_base = 'Tests/SIRDVW_age_OC_2022-05-26_reference_InfectedProvaIGcostante/'
#f_name_base = 'Tests/SIRDVW_age_OC_2022-05-20_reference_Cost1DelayTime3Ages1/'
