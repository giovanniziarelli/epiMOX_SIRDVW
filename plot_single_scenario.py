import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pltlab
import datetime
import numpy as np

f_name = 'Tests/SIRDVW_age_OC_2022-06-21_HospitalizedAll/' 
US1 = pd.read_hdf(f_name + 'US1.h5').to_numpy()
US2 = pd.read_hdf(f_name + 'US2.h5').to_numpy()
UR = pd.read_hdf(f_name + 'UR.h5').to_numpy()

day_init = pd.to_datetime('2021-01-01')
day_init_vaccines = day_init
Tf_data = '2021-05-27'
vaccines = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccines_age.csv') 
vaccines['data'] = pd.to_datetime(vaccines.data) 
vaccines.set_index(['data', 'eta'],inplace=True) 
vaccines.fillna(0,inplace=True) 
vaccines_init = vaccines[:day_init_vaccines-pd.Timedelta(1,'day')].sum() 
vaccines = vaccines.loc[day_init_vaccines:Tf_data] 
length = 147 #105
I = np.zeros((length, int(length/7)))
for i in range(int(length/7)):
    I[i*7:(i+1)*7, i] = 1
US1_t = np.reshape(vaccines['prima_dose'].to_numpy(), (9, length), order = 'F')
US2_t = np.reshape(vaccines['seconda_dose'].to_numpy() + vaccines['mono_dose'].to_numpy(), (9, length), order = 'F')
UR_t  = np.reshape(vaccines['pregressa_infezione'].to_numpy(), (9, length ), order = 'F')

US1_r = np.zeros((6, length))
US1_r[0] = US1_t[0]
US1_r[1] = np.sum(US1_t[1:3], axis = 0)
US1_r[2] = np.sum(US1_t[3:5], axis = 0)
US1_r[3] = np.sum(US1_t[5:7], axis = 0)
US1_r[4] = US1_t[7]
US1_r[5] = US1_t[8]
Ns = 5
ages_opt = [0,1,2,3,4]
delay_time = 3
ncolreduct = int(length/7)

US1_r = np.append(US1_r[:4,:], np.add.reduce(US1_r[4:,:], axis = 0).reshape(1, length), axis = 0)

Nc = 6 #Compartments 
codes = ['0-19', '20-39', '40-59', '60-79', '80+']

day_init_2 = pd.to_datetime('2021-01-01')
colors = plt.cm.cool(np.linspace(0, 1, 5))
date_str = [day_init + datetime.timedelta(k) for k in range(length)]
fig,axes1 = plt.subplots(1,2, figsize = (10, 5))
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
for k in range(5):
    perc_DPC = US1_r[k] / US1_r_tot
    if k == 4:
        perc_DPC[:-1] = 1 - baseline
    axes1[0].bar(weeks_list,perc_DPC[:-1], bottom = baseline,width = 5, color  = colors[k])
    baseline = baseline + perc_DPC[:-1]
plt.ylim(0,1)
axes1[0].set_title('DPC data')
axes1[0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))
baseline = 0.0 * np.ones(weeks-1)
US1_tot = np.sum(US1, axis = 0)
for i in range(len(US1_tot)):
    if US1_tot[i] == 0:
        US1_tot[i] == 1
for k in range(5):
    perc = US1[k] / US1_tot
    axes1[1].bar(weeks_list,perc[:-1], bottom = baseline,width = 5, color = colors[k])
    baseline = baseline + perc[:-1]
axes1[1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))
axes1[1].set_title('Optimal policy')
fig.legend(ages)
plt.ylim(0,1)
fig.suptitle('Percentage of first doses', fontsize = 16)
plt.show()
plt.savefig(f_name + 'img/US1_perc_new.png')


fig5, ax5 = plt.subplots(2,3, figsize=(30,10))
plt.xticks(fontsize = 12)
plt.yticks(fontsize=12)

ax5[0,0].plot(dates_list, np.repeat(US1[0,:-1],7), color = colors[1])
ax5[0,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[0,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,0].plot(dates_list, np.repeat(US1_r[0,:-1], 7), '--', color = 'orange')
ax5[0,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[0,:-1],7), color = 'orange', alpha = 0.08)
ax5[0,0].set_title('0-19')
ax5[0,0].grid()
ax5[0,0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[0,1].plot(dates_list, np.repeat(US1[1,:-1], 7), color = colors[1])
ax5[0,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[1,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,1].plot(dates_list, np.repeat(US1_r[1,:-1], 7), '--', color = 'orange')
ax5[0,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[1,:-1],7), color = 'orange', alpha = 0.08)
ax5[0,1].set_title('20-39')
ax5[0,1].grid()
ax5[0,1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[0,2].plot(dates_list, np.repeat(US1[2,:-1], 7), color = colors[1])
ax5[0,2].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[2,:-1],7), color = colors[1], alpha = 0.08)
ax5[0,2].plot(dates_list, np.repeat(US1_r[2,:-1],7), '--', color = 'orange')
ax5[0,2].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[2,:-1],7), color = 'orange', alpha = 0.08)
ax5[0,2].set_title('40-59')
ax5[0,2].grid()
ax5[0,2].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,0].plot(dates_list, np.repeat(US1[3,:-1], 7), color = colors[1])
ax5[1,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[3,:-1],7), color = colors[1], alpha = 0.08)
ax5[1,0].plot(dates_list, np.repeat(US1_r[3,:-1],7), '--', color = 'orange')
ax5[1,0].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[3,:-1],7), color = 'orange', alpha = 0.08)
ax5[1,0].set_title('60-79')
ax5[1,0].grid()
ax5[1,0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,1].plot(dates_list, np.repeat(US1[4,:-1], 7), color = colors[1])
ax5[1,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1[4,:-1],7), color = colors[1], alpha = 0.08)
ax5[1,1].plot(dates_list, np.repeat(US1_r[4,:-1],7), '--', color = 'orange')
ax5[1,1].fill_between(dates_list, np.zeros(len(dates_list)), np.repeat(US1_r[4,:-1],7), color = 'orange', alpha = 0.08)
ax5[1,1].set_title('80+')
ax5[1,1].grid()
ax5[1,1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,2].remove()

fig5.legend(['Optimal policy', 'Initial policy'])
plt.suptitle('US1')
plt.savefig(f_name+'/img/US1_mod_pp_new.png')
plt.show()


plt.close('all')

f_name_base = 'Tests/SIRDVW_age_OC_2022-05-26_reference_InfectedProvaIGcostante/'

simdf_opt = pd.read_hdf(f_name + 'simdf.h5')
simdf_base = pd.read_hdf(f_name_base + 'simdf.h5')

fig6, ax6 = plt.subplots(1,1, figsize = (5,5))
simdf_opt_sum = simdf_opt.groupby('date').sum()
simdf_base_sum = simdf_base.groupby('date').sum()

ax6.plot(dates_list,simdf_opt_sum.Infected[:-7], color = colors[1])
ax6.plot(dates_list, simdf_base_sum.Infected[:-7], '*', color = 'orange')
ax6.fill_between(dates_list, simdf_opt_sum.Infected[:-7], simdf_base_sum.Infected[:-7], color = 'green', alpha = 0.18)
ax6.grid()
ax6.set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

fig6.legend(['Optimal policy', 'Initial policy'])
plt.suptitle('Infected')
plt.savefig(f_name+'/img/Infected_pp_new.png')
plt.show()

colors_2 = plt.cm.cool(np.linspace(0, 1, 6))
fig8, ax8 = plt.subplots(1,2, figsize = (10,5))
baseline_o = np.zeros(len(dates_list))
baseline_b = np.zeros(len(dates_list))
for i, age in enumerate(['0-19', '20-39', '40-59', '60-79', '80+']):
    ax8[0].plot(dates_list, simdf_opt[simdf_opt.Age == age].Infected[:-7].to_numpy()/simdf_opt_sum.Infected[:-7].to_numpy() + baseline_o, color = colors[i])
    ax8[0].fill_between(dates_list, simdf_opt[simdf_opt.Age == age].Infected[:-7].to_numpy()/simdf_opt_sum.Infected[:-7].to_numpy() + baseline_o, baseline_o, color = colors_2[i+1], alpha = 0.5)
    ax8[1].plot(dates_list, simdf_base[simdf_base.Age == age].Infected[:-7].to_numpy()/simdf_base_sum.Infected[:-7].to_numpy() + baseline_b, color = colors[i], ls = 'dashed')
    ax8[1].fill_between(dates_list, simdf_base[simdf_base.Age == age].Infected[:-7].to_numpy()/simdf_base_sum.Infected[:-7].to_numpy() + baseline_b, baseline_b, color = colors_2[i+1], alpha = 0.5)
    baseline_o += simdf_opt[simdf_opt.Age == age].Infected[:-7].to_numpy()/simdf_opt_sum.Infected[:-7].to_numpy() 
    baseline_b += simdf_base[simdf_opt.Age == age].Infected[:-7].to_numpy()/simdf_base_sum.Infected[:-7].to_numpy() 
ax8[0].grid()
ax8[0].set_title('Optimal policy')
ax8[0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax8[1].grid()
ax8[1].set_title('Initial policy')
ax8[1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))
plt.savefig(f_name+'/img/Infected_perc_pp_new.png')
plt.show()

fig7, ax7 = plt.subplots(1,1, figsize = (5,5))
ax7.plot(dates_list,simdf_opt_sum.Deceased[:-7], color = colors[1])
ax7.plot(dates_list, simdf_base_sum.Deceased[:-7], '*', color = 'orange')
ax7.fill_between(dates_list, simdf_opt_sum.Deceased[:-7], simdf_base_sum.Deceased[:-7], color = 'green', alpha = 0.18)
ax7.grid()
ax7.set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

fig7.legend(['Optimal policy', 'Initial policy'])
plt.suptitle('Deceased')
plt.savefig(f_name+'/img/Deceased_pp_new.png')
plt.show()

fig9, ax9 = plt.subplots(1,2, figsize = (10,5))
baseline_o = np.zeros(len(dates_list))
baseline_b = np.zeros(len(dates_list))
for i, age in enumerate(['0-19', '20-39', '40-59', '60-79', '80+']):
    ax9[0].plot(dates_list, simdf_opt[simdf_opt.Age == age].Deceased[:-7].to_numpy()/simdf_opt_sum.Deceased[:-7].to_numpy() + baseline_o, color = colors[i])
    ax9[0].fill_between(dates_list, simdf_opt[simdf_opt.Age == age].Deceased[:-7].to_numpy()/simdf_opt_sum.Deceased[:-7].to_numpy() + baseline_o, baseline_o, color = colors_2[i+1], alpha = 0.5)
    ax9[1].plot(dates_list, simdf_base[simdf_base.Age == age].Deceased[:-7].to_numpy()/simdf_base_sum.Deceased[:-7].to_numpy() + baseline_b, color = colors[i], ls = 'dashed')
    ax9[1].fill_between(dates_list, simdf_base[simdf_base.Age == age].Deceased[:-7].to_numpy()/simdf_base_sum.Deceased[:-7].to_numpy() + baseline_b, baseline_b, color = colors_2[i+1], alpha = 0.5)
    baseline_o += simdf_opt[simdf_opt.Age == age].Deceased[:-7].to_numpy()/simdf_opt_sum.Deceased[:-7].to_numpy() 
    baseline_b += simdf_base[simdf_opt.Age == age].Deceased[:-7].to_numpy()/simdf_base_sum.Deceased[:-7].to_numpy() 
ax9[0].grid()
ax9[0].set_title('Optimal policy')
ax9[0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax9[1].grid()
ax9[1].set_title('Initial policy')
ax9[1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))
plt.savefig(f_name+'/img/Deceased_perc_pp_new.png')
plt.show()
