import matplotlib.pyplot as plt
import numpy as np
import os

plt.close('all')

#%% loading experimental and simulated data

exp_path = 'C:/Users/maxog/Desktop/Diplomovka_folder/SW/virtual_lockin/data_to_compare/'
exp_data_name = 'FS_2023-01-12_06-17-27_G1R_0.1A_10kOhm_20dB.txt'
sim_data_name = 'simulation.txt'

sim_data = np.loadtxt(os.path.join(exp_path, sim_data_name), skiprows = 1)
exp_data = np.loadtxt(os.path.join(exp_path, exp_data_name), skiprows = 1)

#%% comparison of experiment and simulation
t_exp = exp_data[:,1]
x_exp = exp_data[:,2]
y_exp = exp_data[:,3]

t_sim = sim_data[:,0]
x_sim = sim_data[:,1]
y_sim = sim_data[:,2]

sim_max = max(x_sim)
exp_max = max(x_exp)


plt.figure('Simulation')
plt.plot(t_exp, x_exp/10, label ='Real component exp /10', color = 'black')
plt.plot(t_exp, y_exp/10, label ='Imag component exp/10', color = 'red')
plt.plot(t_sim, x_sim-10.5e-6, label ='Real component', color = 'black', linestyle = 'dashed')
plt.plot(t_sim, y_sim-10.5e-6, label ='Imag component', color = 'red', linestyle = 'dashed')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Voltage [V]')
plt.show()



