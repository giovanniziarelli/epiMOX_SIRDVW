import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import datetime
from mpl_toolkits import mplot3d
import numpy as np


Rt_0  = np.zeros((20,20))
Rt_28 = np.zeros((20,20))
Rt_56 = np.zeros((20,20))
Rt_84 = np.zeros((20,20))
Rt_112 = np.zeros((20,20))
Rt_140 = np.zeros((20,20))
for i in range(20):
    for j in range(20):
        Rt_temp = pd.read_csv('Tests/SIRDVW_age_2022-07-06_Theta1'+str(i+1)+'Theta2'+str(j+1)+'/Rt.csv').to_numpy()
        Rt_0[i,j]  = Rt_temp[0]
        Rt_28[i,j] = Rt_temp[28]
        Rt_56[i,j] = Rt_temp[56]
        Rt_84[i,j] = Rt_temp[84]
        Rt_112[i,j] = Rt_temp[112]
        Rt_140[i,j] = Rt_temp[140]
theta1 = np.linspace(0.05, 1.0, 20)
theta2 = np.linspace(0.05, 1.0, 20)
print('Rt0', Rt_0)
fig1,ax1 = plt.subplots(1,1, figsize = (8,8))
contours = plt.contour(theta1,theta2,Rt_0,8, colors = 'black', linewidths=1.2)
plt.clabel(contours, inline = True, inline_spacing = 10, fontsize = 7, fmt='%1.3f')

plt.imshow(Rt_0, extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))

plt.colorbar()


ax1.axvline(x = 0.197, color = 'red', linestyle='dashed')
ax1.axhline(y = 0.036, color = 'red', linestyle='dashed')
ax1.plot(0.197, 0.036, 'o', linewidth=0.1, color='red')
ax1.annotate(str(Rt_0[3,0])[:8],xy=(0.06, 0.07), fontsize = 8, color='red')
plt.savefig('Tests/Rt_0_theta.png')
plt.show()

fig2,ax2 = plt.subplots(1,1, figsize = (8,8))
contours = plt.contour(theta1,theta2,Rt_28 ,8, colors = 'black', linewidths=1.2) 
plt.clabel(contours, inline = True, inline_spacing = 10, fontsize = 7, fmt='%1.3f')
plt.imshow(Rt_28- Rt_28[3,0], extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
plt.colorbar(format = mticker.FuncFormatter(g))
ax2.axvline(x = 0.197, color = 'red', linestyle='dashed')
ax2.axhline(y = 0.036, color = 'red', linestyle='dashed')
ax2.plot(0.197, 0.036, 'o', linewidth=0.1, color='red')
ax2.annotate(str(Rt_28[3,0])[:8],xy=(0.06, 0.10), fontsize = 8, color='red')
plt.savefig('Tests/Rt_28_theta.png')
plt.show()
print('Reference value', str(Rt_28[3,0]))

fig3,ax3 = plt.subplots(1,1, figsize = (8,8))
contours = plt.contour(theta1,theta2,Rt_56 ,8, colors = 'black', linewidths=1.2)
plt.clabel(contours, inline = True, inline_spacing = 10, fontsize = 7, fmt='%1.3f')
plt.imshow(Rt_56- Rt_56[3,0], extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
plt.colorbar(format = mticker.FuncFormatter(g))
ax3.axvline(x = 0.197, color = 'red', linestyle='dashed')
ax3.axhline(y = 0.036, color = 'red', linestyle='dashed')
ax3.plot(0.197, 0.036, 'o', linewidth=0.1, color='red')
ax3.annotate(str(Rt_56[3,0])[:8],xy=(0.06, 0.07), fontsize = 8, color='red')
plt.savefig('Tests/Rt_56_theta.png')
plt.show()
print('Reference value', str(Rt_56[3,0]))

fig4,ax4 = plt.subplots(1,1, figsize = (8,8))
contours = plt.contour(theta1,theta2,Rt_84,8, colors = 'black', linewidths=1.2)
plt.clabel(contours, inline = True, inline_spacing = 10, fontsize = 7, fmt='%1.3f')
plt.imshow(Rt_84- Rt_84[3,0], extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
plt.colorbar(format = mticker.FuncFormatter(g))
ax4.axvline(x = 0.197, color = 'red', linestyle='dashed')
ax4.axhline(y = 0.036, color = 'red', linestyle='dashed')
ax4.plot(0.197, 0.036, 'o', linewidth=0.1, color='red')
ax4.annotate(str(Rt_84[3,0])[:8],xy=(0.22, 0.05), fontsize = 8, color='red')
plt.savefig('Tests/Rt_84_theta.png')
plt.show()
print('Reference value', str(Rt_84[3,0]))

fig5,ax5 = plt.subplots(1,1, figsize = (8,8))
contours = plt.contour(theta1,theta2,Rt_112,8, colors = 'black', linewidths=1.2)
plt.clabel(contours, inline = True, inline_spacing = 10, fontsize = 7, fmt='%1.3f')
plt.imshow(Rt_112- Rt_112[3,0], extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
plt.colorbar(format = mticker.FuncFormatter(g))
ax5.axvline(x = 0.197, color = 'red', linestyle='dashed')
ax5.axhline(y = 0.036, color = 'red', linestyle='dashed')
ax5.plot(0.197, 0.036, 'o', linewidth=0.1, color='red')
ax5.annotate(str(Rt_112[3,0])[:8],xy=(0.06, 0.06), fontsize = 8, color='red')
plt.savefig('Tests/Rt_112_theta.png')
plt.show()
print('Reference value', str(Rt_112[3,0]))

fig6,ax6 = plt.subplots(1,1, figsize = (8,8))
contours = plt.contour(theta1,theta2,Rt_140 ,8, colors = 'black', linewidths=1.2)
plt.clabel(contours, inline = True, inline_spacing = 10, fontsize = 7, fmt='%1.3f')
plt.imshow(Rt_140- Rt_140[3,0], extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
plt.colorbar(format = mticker.FuncFormatter(g))
ax6.axvline(x = 0.197, color = 'red', linestyle='dashed')
ax6.axhline(y = 0.036, color = 'red', linestyle='dashed')
ax6.plot(0.197, 0.036, 'o', linewidth=0.1, color='red')
ax6.annotate(str(Rt_140[3,0])[:8],xy=(0.22, 0.05), fontsize = 8, color='red')
plt.savefig('Tests/Rt_140_theta.png')
plt.show()
print('Reference value', str(Rt_140[3,0]))

### print('Rt28', Rt_28)
### print('Rt56', Rt_56)
### print('Rt84', Rt_84)
### fig1,ax1 = plt.subplots(1,1, figsize = (8,8))
### contours = plt.contour(theta1,theta2,Rt_0,5, colors = 'black')
### plt.clabel(contours, inline = True, fontsize = 7)

### plt.imshow(Rt_0, extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
### plt.colorbar()
### plt.savefig('Tests/Rt_0_theta.png')
### plt.show()
### fig2,ax2 = plt.subplots(1,1, figsize = (8,8))
### contours = plt.contour(theta1,theta2,Rt_28,5, colors = 'black')
### plt.clabel(contours, inline = True, fontsize = 7)

### plt.imshow(Rt_28, extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
### plt.colorbar()
### plt.savefig('Tests/Rt_28_theta.png')
### plt.show()
### fig3,ax3 = plt.subplots(1,1, figsize = (8,8))
### contours = plt.contour(theta1,theta2,Rt_56,5, colors = 'black')
### plt.clabel(contours, inline = True, fontsize = 7)

### plt.imshow(Rt_56, extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
### plt.colorbar()
### plt.savefig('Tests/Rt_56_theta.png')
### plt.show()
### fig4,ax4 = plt.subplots(1,1, figsize = (8,8))
### contours = plt.contour(theta1,theta2,Rt_84,5, colors = 'black')
### plt.clabel(contours, inline = True, fontsize = 7)

### plt.imshow(Rt_84, extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
### plt.colorbar()
### plt.savefig('Tests/Rt_84_theta.png')
### plt.show()
### fig5,ax5 = plt.subplots(1,1, figsize = (8,8))
### contours = plt.contour(theta1,theta2,Rt_112,5, colors = 'black')
### plt.clabel(contours, inline = True, fontsize = 7)

### plt.imshow(Rt_112, extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
### plt.colorbar()
### plt.savefig('Tests/Rt_112_theta.png')
### plt.show()
### fig6,ax6 = plt.subplots(1,1, figsize = (8,8))
### contours = plt.contour(theta1,theta2,Rt_140,5, colors = 'black')
### plt.clabel(contours, inline = True, fontsize = 7)

### plt.imshow(Rt_140, extent = [0,1,0,1],origin = 'lower', cmap = 'BuPu', alpha = 0.5)
### plt.colorbar()
### plt.savefig('Tests/Rt_140_theta.png')
### plt.show()
### cs = ax1.contourf(sigma1,sigma2,Rt_0)
### ax1.contour(cs, colors = 'k', linestyle = 'dashdot')
### #ax1.plot3D(sigma1,sigma2,Rt_0)
### ax1.set_xlabel('SigmaV')
### ax1.set_ylabel('SigmaW')
### ax1.set_title('Rt at t = 0')
### plt.show()

### fig2,ax2 = plt.subplots(1,1, figsize = (10,10))
### ax2.contour(sigma1,sigma2,Rt_28)
### ax2.set_xlabel('SigmaV')
### ax2.set_ylabel('SigmaW')
### ax2.set_title('Rt at t = 28')
### plt.show()

### fig3,ax3 = plt.subplots(1,1, figsize = (10,10))
### ax3.contour(sigma1,sigma2,Rt_56)
### ax3.set_xlabel('SigmaV')
### ax3.set_ylabel('SigmaW')
### ax3.set_title('Rt at t = 56')
### plt.show()


### fig4,ax4 = plt.subplots(1,1, figsize = (10,10))
### ax4.contour(sigma1,sigma2,Rt_84)
### ax4.set_xlabel('SigmaV')
### ax4.set_ylabel('SigmaW')
### ax4.set_title('Rt at t = 84')
### plt.show()

