import numpy as np
import matplotlib.pyplot as plt
import NonlinearFitting
import time
from data_load import load_soarted_data
from data_manipulation import  MagField, ElCurrent, DrivingForce, FindAmplitude, FindAmplitudeNoBgnd, SubtractBackground, WireVelocity
from plots import single_plot, all_plots, derived_params_plots
import seaborn as sns
from tqdm import tqdm 

#%%
plt.close('all')

#%% parameters of measurement 
      
wire_types = ['G1L', 'G1R', 'G3R']  
temperatures = ['920', '700', '500', '250', 'baseT']  #mK
magnetic_field_current = ['0.1', '0.075', '0.5']        #A
type_of_measurement = ['Vacuum', 'Helium']                        
keys =  ['measurement', 'magnet_drive', 'temperature', 'wire','data', 'drive']

resistance = 10e3 #Ohm
amplif = 100
attenuation = 0.1
wire_length = 1e-3 #m
          
datasets = load_soarted_data()

amplitudes = []

#%%
voltages = []
velocities = []
forces = []
currents = []

typeM = 'Vacuum'
T = temperatures[4]
wire = 'G3R'
magnet_drive = '0.1'
# # derived_params_plots('Vacuum')
             
for dic in datasets:                      
    if (dic.get('measurement') == typeM) and  (dic.get('temperature') == T) and (dic.get('wire') == wire) and dic.get('magnet_drive') == magnet_drive:
                        
        V = np.array(SubtractBackground(dic.get('data')[:,2])) / amplif
        B = MagField(float(dic.get('magnet_drive')))     
        drive = dic.get('drive')
        # print(dic.get('temperature'), drive)
           
        I = ElCurrent(drive, resistance, attenuation)
        F = DrivingForce(B, wire_length, I)
        V0 = FindAmplitudeNoBgnd(V)
        voltages.append(V0)  
        currents.append(I)
        
        forces.append(F)
        # print(len(forces))
        velocities.append(WireVelocity(V0, B, wire_length))  

# =============================================================================
# # break
# fig_name = wire + ' Induced Voltage vs Driving Current'
# plt.figure(fig_name)
# plt.xlabel('Driving Current [A]')
# plt.ylabel('Induced Voltage [V]')
# plt.plot(currents, voltages, marker = 'o', linestyle = '--', label = f' {T}')
# plt.legend()
# 
# fig_name = wire + ' Driving Force vs Peak velocity'
# plt.figure(fig_name)
# plt.xlabel('Driving Force [N]')
# plt.ylabel('Peak velocity [ms-1]')
# plt.plot(forces, velocities, marker = 'o', linestyle = '--', label = f' {T}')
# 
# plt.xscale('log') 
# plt.yscale('log')
# plt.legend()
# plt.show()
# =============================================================================

# derived_params_plots()



#%% fitting nonlinear data
fit = False
if fit:
    all_tuples = []
    T = temperatures[4]
    print('Temperature ', T, ' mK')
    wire = 'G3R'
    typeM = 'Helium'
    magnet_drive = '0.075'
      
    
    for dic in datasets:
        if (dic.get('measurement') == typeM) and  (dic.get('temperature') == T) and (dic.get('wire') == wire) and dic.get('magnet_drive') == magnet_drive:      
            freq = dic.get('data')[:,1]
            xs = np.array(SubtractBackground(dic.get('data')[:,2])) / amplif
            ys = np.array(dic.get('data')[:, 3]) /amplif
            plt.plot(freq,xs)  
            drive = dic.get('drive')
            label = f'{typeM},{wire},{T},{drive},{magnet_drive}'
            fit_tuple = (drive, freq, xs, ys, label)
            all_tuples.append(fit_tuple)
          
    all_tuples.sort(key = lambda tup: tup[0])
     
    drives = []
    freq = []
    xs = [] 
    ys = []
    labels = []   
     
    # all_tuples.remove(all_tuples[0])
    # all_tuples.remove(all_tuples[0])
    
    
    t_start = time.time()
    for count,j in enumerate(all_tuples):    
    
        drives.append(all_tuples[count][0])
        freq.append(all_tuples[count][1])
        xs.append(all_tuples[count][2])
        ys.append(all_tuples[count][3])
        labels.append(all_tuples[count][4])
    
    
    if typeM == 'Vacuum':
        fitted_data = NonlinearFitting.GoalpostFit(freq, xs, ys, drives, labels)         
        #fitted_data.remove_transient()
        
        fitted_data.optional_settings(correct_phase = False,peak_weight=100)
        # fitted_data.fit(vary_pars={'g1': True, 'g2':True, 'g3':True},  bounds = {'g2': (0,1e-9)}) 
        #fitted_data.fit(vary_pars={'g1': True, 'g2':False, 'g3':True}, init_values = {'g2':0,'g3':1e12})#, bounds = {'g3': (0,1e3)})     
        #fitted_data.fit(vary_pars={'g1': True, 'g2':True, 'g3':False}, init_values = {'g3':0 },  bounds = {'g2': (0,1e3)})     
        fitted_data.fit(vary_pars={'g1': True, 'g2':False, 'g3':False}, init_values = {'g2':0 ,'g3':0})#, bounds = {'g1': (0,1)})     
    
    
    else:
        fitted_data = NonlinearFitting.NbTiFit(freq, xs, ys, drives, labels)
        fitted_data.optional_settings(correct_phase = False,peak_weight=100,show_weights=False)
        # fitted_data.fit(vary_pars={'g1': True, 'g2':True, 'g3':True},  bounds = {'g2': (0,1e-9)}) 
        #fitted_data.fit(vary_pars={'g1': True, 'g2':False, 'g3':True}, init_values = {'g2':0,'g3':1e12})#, bounds = {'g3': (0,1e3)})     
        #fitted_data.fit(vary_pars={'g1': True, 'g2':True, 'g3':False}, init_values = {'g3':0 },  bounds = {'g2': (0,1e3)})     
        fitted_data.fit(vary_pars={'g1': True, 'g2':False, 'g3':False}, init_values = {'g2':0 ,'g3':0})#, bounds = {'g1': (0,1)})     
    
    
    t_end = time.time()
    print('Fit took',t_end - t_start,'seconds')        

    fig_path =".\\IMG"
    fitted_data.figures(path = fig_path ,extension='.png', show=True, names = labels,resolution=(800,800))
    print('Figures took',t_end - t_start,'seconds')
    fitted_data.save_params(folder='output_textfiles', name = labels[0])
    fitted_data.save_reports(folder='output_results', name = labels)
    
    
#%%
plt.show()

