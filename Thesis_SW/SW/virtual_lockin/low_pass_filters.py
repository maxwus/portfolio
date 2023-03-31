# from scipy import butter, freqz
import numpy as np
import matplotlib.pyplot as plt


def rc_lpf(cut_off, order):
    '''

    '''
    def filt(f):
        return (1.0 / (( f / cut_off) * 1.0j + 1.0))**order

    return filt


def rc_analog_filter(frqs, R1 ,C1, R2 = 1, C2 = 1, R3 = 1, C3 = 1, R4 = 1 , C4 = 1,  order = 1, show = False):
    
    
    if order == 1:
        cut_off = 1/(2*np.pi*R1*C1)
          
    elif order == 2:
        cut_off = 1/2/np.pi/np.sqrt(R1*R2*C1*C2)
    
    elif order == 3:
        cut_off = 1/2/np.pi/np.sqrt(R1*R2*R3*C1*C2*C3)
    
    elif order == 4:
        cut_off = 1/2/np.pi/np.sqrt(R1*R2*R3*R4*C1*C2*C3*C4)
    
    else:
        raise ValueError("Use RC filter of order 1-4")
    

    V_in = 1
    V_out = np.abs(V_in / (( frqs / cut_off) *1j + 1))
    
    if show:
        plt.figure('Response of analog RC Filter')
        plt.plot(frqs, 20*np.log10(abs(V_out)), label = f' Rc filter of order {order} and f_cut {cut_off}')       
        # plt.axvline(cut_off, color='k')
        plt.xscale('log')  
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Amp attenuation [dB]')
        plt.legend()
        plt.grid(True)
    
    






