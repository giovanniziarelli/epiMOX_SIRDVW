import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pltlab
import datetime
import numpy as np

#f_name_IC1 = 'Tests/SIRDVW_age_OC_2022-09-19_HOMO_Const2_Cost1_R01/' 
#f_name_IC2 = 'Tests/SIRDVW_age_OC_2022-09-19_HOMO_Const2_Cost1_R02/'
#f_name_IC3 = 'Tests/SIRDVW_age_OC_2022-09-19_HOMO_Const2_Cost1_R03/'

#f_name_IC1 = 'Tests/SIRDVW_age_OC_2022-09-21_HOMO_Const3_Cost3_R01/' 
#f_name_IC2 = 'Tests/SIRDVW_age_OC_2022-09-21_HOMO_Const3_Cost3_R02/'
#f_name_IC3 = 'Tests/SIRDVW_age_OC_2022-09-21_HOMO_Const3_Cost3_R03/'

f_name_IC1 = 'Tests/SIRDVW_age_OC_2022-09-23_onda_quadra_Cost3_R01/' 
f_name_IC2 = 'Tests/SIRDVW_age_OC_2022-09-23_onda_quadra_Cost3_R02/'
f_name_IC3 = 'Tests/SIRDVW_age_OC_2022-09-23_onda_quadra_Cost3_R03/'

#f_name_base_IC1 = 'Tests/SIRDVW_age_2022-09-19_R0_1_Cost1/'
f_name_base_IC1 = 'Tests/SIRDVW_age_2022-09-23_onda_quadra_R01/'

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
Ns = 5
ages_opt = [0,1,2,3,4]
delay_time = 3
ncolreduct = int(length/7)
###############################################################################RIMETTERE NCOLREDUCT AL POSTO LENGTH

US1_IC1 = 0.5 * pd.read_hdf(f_name_IC1 + 'US1.h5').to_numpy()
US1_IC2 = 0.5 * pd.read_hdf(f_name_IC2 + 'US1.h5').to_numpy()
US1_IC3 = 0.5 * pd.read_hdf(f_name_IC3 + 'US1.h5').to_numpy()

US1_IC1_ig = 0.5 * pd.read_hdf(f_name_base_IC1 + 'US1.h5').to_numpy()
#US1_IC1_ig[:, :3] = US1_IC1_ig[:, 3:6] no per onda quadra
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
length2 = 98 
dates_list = [day_init + datetime.timedelta(x) for x in range(length2)]

weeks_list = [day_init + datetime.timedelta(7*x) for x in range(int(length2/7))]
colors = plt.cm.cool(np.linspace(0, 1, 5))

fig5, ax5 = plt.subplots(2,3, figsize=(30,10))
plt.xticks(fontsize = 12)
plt.yticks(fontsize=12)

ax5[0,0].plot(dates_list, np.repeat(US1_IC1[0,:-1],7), color = colors[1])

ax5[0,0].plot(dates_list, np.repeat(US1_IC2[0,:-1],7), color = colors[4])

ax5[0,0].plot(dates_list, np.repeat(US1_IC3[0,:-1],7), color = 'salmon')

ax5[0,0].plot(dates_list, np.repeat(US1_IC1_ig[0,:-1], 7), '--', color = 'black')

ax5[0,0].set_title('0-19')
ax5[0,0].grid()

ax5[0,0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[0,1].plot(dates_list, np.repeat(US1_IC1[1,:-1], 7), color = colors[1])

ax5[0,1].plot(dates_list, np.repeat(US1_IC2[1,:-1], 7), color = colors[4])

ax5[0,1].plot(dates_list, np.repeat(US1_IC3[1,:-1], 7), color = 'salmon')

ax5[0,1].plot(dates_list, np.repeat(US1_IC1_ig[1,:-1], 7), '--', color = 'black')

ax5[0,1].set_title('20-39')
ax5[0,1].grid()
ax5[0,1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[0,2].plot(dates_list, np.repeat(US1_IC1[2,:-1], 7), color = colors[1])

ax5[0,2].plot(dates_list, np.repeat(US1_IC2[2,:-1], 7), color = colors[4])

ax5[0,2].plot(dates_list, np.repeat(US1_IC3[2,:-1], 7), color = 'salmon')

ax5[0,2].plot(dates_list, np.repeat(US1_IC1_ig[2,:-1],7), '--', color = 'black')

ax5[0,2].set_title('40-59')
ax5[0,2].grid()
ax5[0,2].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,0].plot(dates_list, np.repeat(US1_IC1[3,:-1], 7), color = colors[1])

ax5[1,0].plot(dates_list, np.repeat(US1_IC2[3,:-1], 7), color = colors[4])

ax5[1,0].plot(dates_list, np.repeat(US1_IC3[3,:-1], 7), color = 'salmon')

ax5[1,0].plot(dates_list, np.repeat(US1_IC1_ig[3,:-1],7), '--', color = 'black')

ax5[1,0].set_title('60-79')
ax5[1,0].grid()
ax5[1,0].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,1].plot(dates_list, np.repeat(US1_IC1[4,:-1], 7), color = colors[1])

ax5[1,1].plot(dates_list, np.repeat(US1_IC2[4,:-1], 7), color = colors[4])

ax5[1,1].plot(dates_list, np.repeat(US1_IC3[4,:-1], 7), color = 'salmon')

ax5[1,1].plot(dates_list, np.repeat(US1_IC1_ig[4,:-1],7), '--', color = 'black')

ax5[1,1].set_title('80+')
ax5[1,1].grid()
ax5[1,1].set_xticks((day_init + datetime.timedelta(0), day_init + datetime.timedelta(28), day_init + datetime.timedelta(56), day_init + datetime.timedelta(84)))

ax5[1,2].remove()

fig5.legend(['Optimal policy - R0 = 0.72', 'Optimal policy - R0 = 1', 'Optimal policy - R0 = 1.30','Initial policy'], bbox_to_anchor=(0.75, 0.25), loc = 'center')
#plt.suptitle('US1')
plt.savefig(f_name_IC1+'/img/US1_COMPARE.png')
plt.show()


plt.close('all')

#f_name_base = 'Tests/SIRDVW_age_OC_2022-05-26_reference_InfectedProvaIGcostante/'
#f_name_base = 'Tests/SIRDVW_age_OC_2022-05-20_reference_Cost1DelayTime3Ages1/'
