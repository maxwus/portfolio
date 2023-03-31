import numpy as np
import matplotlib.pyplot as plt
import simulated_signals as signals
import digital_lockin as dl
from tqdm import tqdm 

plt.close('all')

save_graphs = False
show_raw_signal = False

#%% 
#initialization of  Lock-in parameters 

time_constants = [30 ,10 ,3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003,0.0001]  #availible time_constants

N_points = 60
f_start =  5.535450000000E+3
f_stop =  5.535850000000E+3
order = 3
time_constant = 10.0

#%%
#initializing parameters for simulated signal

# amp = 1.550306476014E-5
f0 = 5.535680000000E+3
# df = 7.89e-3/748684.6105542366
df = 0.024
gamma = df*np.pi*2

rho_Si = 2330
l_beam = 1.0e-3
# l_leg = 5.0e-4 #G3R
l_leg = 10.0e-4 #G1L
width = 22e-6 
thickness = 7e-6

V_beam = l_beam * width * thickness
V_leg = l_leg * width * thickness

m_beam = V_beam * rho_Si
m_leg = V_leg * rho_Si

m = m_beam + 2 * 1/4 * m_leg


drive_V = 0.215
drive_I = drive_V / 10.0 / 10000.0 
B = 0.0126
drive = np.sqrt(2) * B * drive_I * l_beam

print(drive)

#drive = 1.7819090885901002e-11 # force in Newtons

#G1R sim 3x larger
#G1L sim 10x larger
#G3R sim 0.1 larger


# setting time sampling

time_stop = 80
dt = 40e-6
print(f'Points per period: {1/(f0*dt):.2f}')
time_duration = min(time_stop, 10 * time_constant)
ts = np.linspace(0.0, time_stop, int(time_stop/dt) + 1)

# setting frequencies
fs = np.linspace(f_start, f_stop, N_points)       
fs_plot = np.linspace(f_start, f_stop, 1000)       



#%% simulating measurement
output_x = []
output_y = []

final_amp = 0.0
final_phase = 0.0
phase = 0.0
signals_v = []
times = []
drives = []

for n,freq in tqdm(enumerate(fs), total = len(fs)):
     
    if n == 0:
        drive_complex = drive
    
    else:
        # changing phase and and drive to ensure smoothenes between changing frequencies
        phase = final_phase - 2*np.pi*freq*n*time_stop          # absolute phase of the drive
        drive_complex = drive*np.exp(1j*phase)
        ts = np.linspace((n+1)*time_stop-time_duration, (n+1)*time_stop, int(time_duration/dt) + 1)
        
       
    amp_p, vel_p = signals.LHO_part(f0, drive_complex, gamma, m)
    IC = final_amp - amp_p(freq, n*time_stop)
    amp_h, vel_h = signals.LHO_homo(IC, n*time_stop, f0, gamma)  #generating amplitude and velocity of homogenic solution of LHO
    
    # full solution is the sum of particular and  homogenic solution 
    # homogenic solution decyes before the wait time ends
    signal_x = amp_p(freq, ts) + amp_h(ts)
    signal_v = vel_p(freq, ts) + vel_h(ts)
    
    
    if n<5:
        signals_v.append(signal_v)
        times.append(ts)
        drives.append(drive_complex*np.exp(1.0j*freq*2*np.pi*ts))

    final_amp = signal_x[-1]
    final_phase = 2*np.pi*freq*(n+1)*time_stop + phase

    X, Y = dl.lockin_process(ts, signal_v, freq, 180.0/np.pi*phase, time_constant, order)
    # the signal relaxed to final amplitude, last value taken
    output_x.append(X[-1])
    output_y.append(Y[-1])

    if save_graphs:
        plt.figure(f'{freq}')
        plt.plot(ts,X, label = 'real', color = 'black')
        plt.plot(ts,Y, label = 'imag', color = 'red')
        plt.xlabel('ts')
        plt.savefig(fname = f'Real+Imag_output/{freq}.png')
        plt.close(f'{freq}')

#showing raw signal with time dependence
if show_raw_signal:
    plt.figure('Raw data')
    plt.plot(ts,np.real(signal_x), label = 'real', color = 'black')
    plt.plot(ts,np.imag(signal_x), label = 'imag', color = 'red')
    plt.legend()
    plt.xlabel('ts')

#showing lockin output with expected (particular solution of LHO)
show_resonance_curve = False
if show_resonance_curve:
    plt.figure('Resonance curve')
    plt.plot(fs_plot,np.real(vel_p(fs_plot)), label = 'Resonance real', color = 'black', linestyle = '--')
    plt.plot(fs_plot,np.imag(vel_p(fs_plot)), label = 'Resonance imag', color = 'red', linestyle = '--')
    plt.plot(fs[:len(output_x)],output_x, 'ko', label = 'Lock-in real')
    plt.plot(fs[:len(output_y)],output_y, 'ro', label = 'Lock-in imag')
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signal ')
    plt.savefig(fname = 'Real+Imag/Lockin_output.png')


output_data = [fs[:len(output_x)],output_x, output_y]

#%%
save_output_path = 'C:/Users/maxog/Desktop/Diplomovka_folder/SW/virtual_lockin/data_to_compare'

fs = (np.array(fs[:len(output_x)])).T
x = (np.array(output_x)).T
y = (np.array(output_y)).T
output_data = (fs,x,y)    

output_data = np.array(output_data)

np.savetxt(f'{save_output_path}/simulation.txt',  output_data.T)

plt.show()

# =============================================================================
# signal_graph = plt.figure('Time series signal')
# drive_graph = plt.figure('Time series drive')
# 
# for t,v,d in zip(times, signals_v, drives):
#     signal_graph.gca().plot(t,np.real(v), 'k-')
#     drive_graph.gca().plot(t,np.real(d), 'k-')
# 
# plt.show()
# =============================================================================
    

