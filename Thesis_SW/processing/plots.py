import matplotlib.pyplot as plt
import numpy as np
from data_manipulation import SubtractBackground, FindAmplitude, MagField, ElCurrent, DrivingForce, WireVelocity
from data_load import load_soarted_data

plt.close('all')
def single_plot(type_of_measurement, temperature, wire):
    for dic in datasets:
        if (dic.get('measurement') == type_of_measurement) and  (dic.get('temperature') == temperature) and (dic.get('wire') == wire): 
            freq = dic.get('data')[:,1]
            xs = np.array(SubtractBackground(dic.get('data')[:,2])) / amplif
            ys = np.array(dic.get('data')[:, 3]) /amplif
            plt.figure(f'{wire} {type_of_measurement}{temperature}')
            plt.plot(freq,xs)
            plt.show()
            

def derived_params_plots():
    for wire in wire_types:  
        for typeM in type_of_measurement:
            for T in temperatures:              
                for dic in datasets:
                    if (dic.get('measurement') == typeM) and  (dic.get('temperature') == '920') and (dic.get('wire') == wire):
                        resistance = 10e3 #Ohm
                        amplif = 100
                        attenuation = 0.1
                        wire_length = 1e-3 #m
    
                        voltages = []
                        velocities = []
                        forces = []
                        currents = []
                        
                        V = np.array(SubtractBackground(dic.get('data')[:,2])) / amplif
                        B = MagField(float(dic.get('magnet_drive')))     
                        drive = dic.get('drive')
                       
                        I = ElCurrent(drive, resistance, attenuation)
                        V0 = FindAmplitude(V)
                        voltages.append(V0)  
                        currents.append(I)
                        forces.append(DrivingForce(B, wire_length, I))
                        velocities.append(WireVelocity(V0, B, wire_length))  
                        
                      
                        fig_name = wire + ' Induced Voltage vs Driving Current'
                        plt.figure(fig_name)
                        plt.xlabel('Driving Current [A]')
                        plt.ylabel('Induced Voltage [V]')
                        plt.plot(currents, voltages, marker = 'o', linestyle = '--', label = f' {T}')
                        
            
                        
                        fig_name = wire + ' Driving Force vs Peak velocity'
                        plt.figure(fig_name)
                        plt.xlabel('Driving Force [N]')
                        plt.ylabel('Peak velocity [ms-1]')
                        plt.plot(forces, velocities, marker = 'o', linestyle = '--', label = f' {T}')
    
                plt.xscale('log') 
                plt.yscale('log')
                plt.legend()
                plt.show()




def all_plots(temperature):
    for wire in wire_types:
        for typeM in type_of_measurement:     
            for dic in datasets:
                if (dic.get('measurement') == typeM) and  (dic.get('temperature') == temperature) and (dic.get('wire') == wire) :
                    
                    freq = dic.get('data')[:,1]
                    V = np.array(SubtractBackground(dic.get('data')[:,2])) / amplif
                    B = MagField(float(dic.get('magnet_drive')))     
                    drive = dic.get('drive')
                    
                    fig_name = wire + ' voltage, ' + typeM
                    plt.figure(fig_name)
                    plt.xlabel('Frequency [Hz]')
                    plt.ylabel('Voltage [V]')
                    plt.plot(freq, V, label = f'{typeM}, {temperatures[0]}, {drive}' )
                    plt.legend()
                    # temp = (wire, dic.get('drive'),FindAmplitude(V0))
                    # Amplitudes.append(temp)
                    
                    I = ElCurrent(drive, resistance, attenuation)
                    V0 = FindAmplitude(V)
                    voltages.append(V0)  
                    currents.append(I)
                    forces.append(DrivingForce(B, wire_length, I))
                    velocities.append(WireVelocity(V0, B, wire_length))  
                        
                      
                # fig_name = wire + ' Induced Voltage vs Driving Current'
                # plt.figure(fig_name)
                # plt.xlabel('Driving Current [A]')
                # plt.ylabel('Induced Voltage [V]')
                # plt.plot(currents, voltages, marker = 'o', linestyle = '--', label = f'{typeM}, {temperatures[0]}')
                
                
                # fig_name = wire + ' Driving Force vs Peak velocity'
                # plt.figure(fig_name)
                # plt.xlabel('Driving Force [N]')
                # plt.ylabel('Peak velocity [ms-1]')
                # plt.plot(forces, velocities, marker = 'o', linestyle = '--', label = f'{typeM}, {temperatures[0]}')
     
        # plt.yscale('log')
        # plt.legend()
        # plt.show()
        
        
datasets = load_soarted_data()


wire_types = ['G1L', 'G1R', 'G3R']  
temperatures = ['920', '700', '500', '250', 'baseT']  #mK
magnetic_field_current = ['0.1', '0.075', '0.5']        #A
type_of_measurement = ['Vacuum', 'Helium']                        
keys =  ['measurement', 'magnet_drive', 'temperature', 'wire','data', 'drive']

T = temperatures[0]
wire = wire_types[0]

resistance = 10e3 #Ohm
amplif = 100
attenuation = 0.1
wire_length = 1e-3 #m
          

Amplitudes = []

voltages = []
velocities = []
forces = []
currents = []
