import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pltlab
import datetime
import numpy as np

#CASE 1
#f_name_IC1 = 'Tests/SIRDVW_age_OC_2022-09-19_HOMO_Const2_Cost3_R01/' 
#f_name_IC2 = 'Tests/SIRDVW_age_OC_2022-09-19_HOMO_Const2_Cost3_R02/'
#f_name_IC3 = 'Tests/SIRDVW_age_OC_2022-09-19_HOMO_Const2_Cost3_R03/'

#f_name_base_IC1 = 'Tests/SIRDVW_age_2022-09-19_R0_1_Cost1/'
#f_name_base_IC2 = 'Tests/SIRDVW_age_2022-09-19_R0_2_Cost1/'
#f_name_base_IC3 = 'Tests/SIRDVW_age_2022-09-19_R0_3_Cost1/'

#CASE 2
#f_name_IC1 = 'Tests/SIRDVW_age_OC_2022-09-21_HOMO_Const3_Cost3_R01/'
#f_name_IC2 = 'Tests/SIRDVW_age_OC_2022-09-21_HOMO_Const3_Cost3_R02/'
#f_name_IC3 = 'Tests/SIRDVW_age_OC_2022-09-21_HOMO_Const3_Cost3_R03/'

#f_name_base_IC1 = 'Tests/SIRDVW_age_2022-09-21_R01_HOMO_IFR/'
#f_name_base_IC2 = 'Tests/SIRDVW_age_2022-09-21_R02_HOMO_IFR/'
#f_name_base_IC3 = 'Tests/SIRDVW_age_2022-09-21_R03_HOMO_IFR/'

#CASE 3
f_name_IC1 = 'Tests/SIRDVW_age_OC_2022-09-23_onda_quadra_Cost1_R01/'
f_name_IC2 = 'Tests/SIRDVW_age_OC_2022-09-23_onda_quadra_Cost1_R02/'
f_name_IC3 = 'Tests/SIRDVW_age_OC_2022-09-23_onda_quadra_Cost1_R03/'

f_name_base_IC1 = 'Tests/SIRDVW_age_2022-09-23_onda_quadra_R01/'
f_name_base_IC2 = 'Tests/SIRDVW_age_2022-09-23_onda_quadra_R02/'
f_name_base_IC3 = 'Tests/SIRDVW_age_2022-09-23_onda_quadra_R03/'


plt.rc('legend',fontsize='medium')
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
length = 147
I = np.zeros((length, int(length/7)))
for i in range(int(length/7)):
    I[i*7:(i+1)*7, i] = 1
#print(vaccines['prima_dose'])
Ns = 5
ages_opt = [0,1,2,3,4]
delay_time = 3
ncolreduct = int(length/7)
###############################################################################RIMETTERE NCOLREDUCT AL POSTO LENGTH

Nc = 6 #Compartments 
codes = ['0-19', '20-39', '40-59', '60-79', '80+']

day_init_2 = pd.to_datetime('2021-01-01')
colors = plt.cm.cool(np.linspace(0, 1, 5))
date_str = [day_init + datetime.timedelta(k) for k in range(length)]
#f_name_base = 'Tests/SIRDVW_age_OC_2022-05-26_reference_InfectedProvaIGcostante/'
#f_name_base = 'Tests/SIRDVW_age_OC_2022-05-20_reference_Cost1DelayTime3Ages1/'
length2 = 140
dates_list = [day_init + datetime.timedelta(x) for x in range(length2)]
sigma1 = 0.21
sigma2 = 0.11
theta1 = 0.197
theta2 = 0.036
sigma_bar = 0.31 #average infections severe fraction
muH = 0.276 #average mortality for hospitalized


simdf_IC1 = pd.read_hdf(f_name_IC1 + 'simdf_1_month.h5')
simdf_IC2 = pd.read_hdf(f_name_IC2 + 'simdf_1_month.h5')
simdf_IC3 = pd.read_hdf(f_name_IC3 + 'simdf_1_month.h5')

simdf_IC1_base = pd.read_hdf(f_name_base_IC1 + 'simdf_1_month.h5')
simdf_IC2_base = pd.read_hdf(f_name_base_IC2 + 'simdf_1_month.h5')
simdf_IC3_base = pd.read_hdf(f_name_base_IC3 + 'simdf_1_month.h5')

simdf_IC1_sum = simdf_IC1.groupby('date').sum()
simdf_IC2_sum = simdf_IC2.groupby('date').sum()
simdf_IC3_sum = simdf_IC3.groupby('date').sum()

simdf_IC1_base_sum = simdf_IC1_base.groupby('date').sum()
simdf_IC2_base_sum = simdf_IC2_base.groupby('date').sum()
simdf_IC3_base_sum = simdf_IC3_base.groupby('date').sum()

frac_IC1 = (simdf_IC1_sum.Suscept[:-7] + sigma1 * theta1 * simdf_IC1_sum.VaccinatedFirst[:-7]) / (simdf_IC1_sum.Suscept[:-7] + sigma1 * simdf_IC1_sum.VaccinatedFirst[:-7])
frac_IC2 = (simdf_IC2_sum.Suscept[:-7] + sigma1 * theta1 * simdf_IC2_sum.VaccinatedFirst[:-7]) / (simdf_IC2_sum.Suscept[:-7] + sigma1 * simdf_IC2_sum.VaccinatedFirst[:-7])
frac_IC3 = (simdf_IC3_sum.Suscept[:-7] + sigma1 * theta1 * simdf_IC3_sum.VaccinatedFirst[:-7]) / (simdf_IC3_sum.Suscept[:-7] + sigma1 * simdf_IC3_sum.VaccinatedFirst[:-7])

frac_IC1_base = (simdf_IC1_base_sum.Suscept[:-7] + sigma1 * theta1 * simdf_IC1_base_sum.VaccinatedFirst[:-7]) / (simdf_IC1_base_sum.Suscept[:-7] + sigma1 * simdf_IC1_base_sum.VaccinatedFirst[:-7])
frac_IC2_base = (simdf_IC2_base_sum.Suscept[:-7] + sigma1 * theta1 * simdf_IC2_base_sum.VaccinatedFirst[:-7]) / (simdf_IC2_base_sum.Suscept[:-7] + sigma1 * simdf_IC2_base_sum.VaccinatedFirst[:-7])
frac_IC3_base = (simdf_IC3_base_sum.Suscept[:-7] + sigma1 * theta1 * simdf_IC3_base_sum.VaccinatedFirst[:-7]) / (simdf_IC3_base_sum.Suscept[:-7] + sigma1 * simdf_IC3_base_sum.VaccinatedFirst[:-7])

hosp_IC1 = simdf_IC1_sum.Infected[:-7] * frac_IC1 * sigma_bar
hosp_IC2 = simdf_IC2_sum.Infected[:-7] * frac_IC2 * sigma_bar
hosp_IC3 = simdf_IC3_sum.Infected[:-7] * frac_IC3 * sigma_bar

hosp_IC1_base = simdf_IC1_base_sum.Infected[:-7] * frac_IC1_base * sigma_bar
hosp_IC2_base = simdf_IC2_base_sum.Infected[:-7] * frac_IC2_base * sigma_bar
hosp_IC3_base = simdf_IC3_base_sum.Infected[:-7] * frac_IC3_base * sigma_bar

print('Average Infected new sol - initial policy R0 0.72:', np.sum(simdf_IC1_sum.Infected[:-7] - simdf_IC1_base_sum.Infected[:-7])/len(hosp_IC1))
print('Average Infected new sol - initial policy R0 1:   ', np.sum(simdf_IC2_sum.Infected[:-7] - simdf_IC2_base_sum.Infected[:-7])/len(hosp_IC1))
print('Average Infected new sol - initial policy R0 1.30:', np.sum(simdf_IC3_sum.Infected[:-7] - simdf_IC3_base_sum.Infected[:-7])/len(hosp_IC1))

fig6, ax6 = plt.subplots(1,1, figsize = (8,8))

ax6.plot(dates_list,simdf_IC1_sum.Infected[:-7], color = colors[1])
ax6.plot(dates_list,simdf_IC1_base_sum.Infected[:-7], '--',  color = colors[1])

ax6.plot(dates_list,simdf_IC2_sum.Infected[:-7], color = colors[4])
ax6.plot(dates_list,simdf_IC2_base_sum.Infected[:-7], '--',  color = colors[4])

ax6.plot(dates_list,simdf_IC3_sum.Infected[:-7], color = 'salmon')
ax6.plot(dates_list,simdf_IC3_base_sum.Infected[:-7], '--',  color = 'salmon')

ax6.grid()
ax6.set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140)))
ax6.axvspan(day_init + datetime.timedelta(109),day_init + datetime.timedelta(147), alpha =0.2, color='gold',edgecolor='blue', linewidth=0.5)
plt.axvline(x = day_init + datetime.timedelta(109), color = 'gold', linestyle='dashed')
fig6.legend(['Optimal policy - R0 = 0.72', 'Initial policy - R0 = 0.72', 'Optimal policy - R0 = 1', 'Initial policy - R0 = 1', 'Optimal policy - R0 = 1.30', 'Initial policy - R0 = 1.30'])
plt.savefig(f_name_IC1 +'/img/Infected_COMPARE_3IC_1_month_pp_new.png')
plt.show()

print('Deceased new sol - initial policy R0 0.72:', simdf_IC1_sum.Deceased[-7] - simdf_IC1_base_sum.Deceased[-7])
print('Deceased new sol - initial policy R0 1:   ', simdf_IC2_sum.Deceased[-7] - simdf_IC2_base_sum.Deceased[-7])
print('Deceased new sol - initial policy R0 1.30:', simdf_IC3_sum.Deceased[-7] - simdf_IC3_base_sum.Deceased[-7])

fig7, ax7 = plt.subplots(1,1, figsize = (8,8))

ax7.plot(dates_list,simdf_IC1_sum.Deceased[:-7], color = colors[1])
ax7.plot(dates_list,simdf_IC1_base_sum.Deceased[:-7], '--',  color = colors[1])

ax7.plot(dates_list,simdf_IC2_sum.Deceased[:-7], color = colors[4])
ax7.plot(dates_list,simdf_IC2_base_sum.Deceased[:-7], '--',  color = colors[4])

ax7.plot(dates_list,simdf_IC3_sum.Deceased[:-7], color = 'salmon')
ax7.plot(dates_list,simdf_IC3_base_sum.Deceased[:-7], '--',  color = 'salmon')


#ax7.fill_between(dates_list, simdf_opt_sum.Deceased[:-7], simdf_base_sum.Deceased[:-7], color = 'green', alpha = 0.18)
ax7.grid()
ax7.set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140)))
ax7.axvspan(day_init + datetime.timedelta(109),day_init + datetime.timedelta(147), alpha =0.2, color='gold',edgecolor='blue', linewidth=0.5)
plt.axvline(x = day_init + datetime.timedelta(109), color = 'gold', linestyle='dashed')

fig7.legend(['Optimal policy - R0 = 0.72', 'Initial policy - R0 = 0.72', 'Optimal policy - R0 = 1', 'Initial policy - R0 = 1', 'Optimal policy - R0 = 1.30', 'Initial policy - R0 = 1.30'])
#plt.suptitle('Deceased')
plt.savefig(f_name_IC1+'/img/Deceased_COMPARE_IC3_1_month_pp_new.png')
plt.show()

print('Average Hospitalized new sol - initial policy R0 0.72:', np.sum(hosp_IC1 - hosp_IC1_base)/len(hosp_IC1))
print('Average Hospitalized new sol - initial policy R0 1:   ', np.sum(hosp_IC2 - hosp_IC2_base)/len(hosp_IC1))
print('Average Hospitalized new sol - initial policy R0 1.30:', np.sum(hosp_IC3 - hosp_IC3_base)/len(hosp_IC1))

fig8, ax8 = plt.subplots(1,1, figsize = (8,8))

ax8.plot(dates_list,hosp_IC1, color = colors[1])
ax8.plot(dates_list,hosp_IC1_base, '--',  color = colors[1])

ax8.plot(dates_list,hosp_IC2, color = colors[4])
ax8.plot(dates_list,hosp_IC2_base, '--',  color = colors[4])

ax8.plot(dates_list,hosp_IC3, color = 'salmon')
ax8.plot(dates_list,hosp_IC3_base, '--',  color = 'salmon')

ax8.grid()
ax8.set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84), day_init + datetime.timedelta(112), day_init + datetime.timedelta(140)))
ax8.axvspan(day_init + datetime.timedelta(109),day_init + datetime.timedelta(147), alpha =0.2, color='gold',edgecolor='blue', linewidth=0.5)
plt.axvline(x = day_init + datetime.timedelta(109), color = 'gold', linestyle='dashed')

fig8.legend(['Optimal policy - R0 = 0.72', 'Initial policy - R0 = 0.72', 'Optimal policy - R0 = 1', 'Initial policy - R0 = 1', 'Optimal policy - R0 = 1.30', 'Initial policy - R0 = 1.30'])
#plt.suptitle('Deceased')
plt.savefig(f_name_IC1+'/img/Hospitalized_COMPARE_IC3_1_month_pp_new.png')
plt.show()
#plt.show()

