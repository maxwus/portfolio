import numpy as np
import low_pass_filters as lpf
import matplotlib.pyplot as plt




def lockin_process(ts, sig, ref_freq, ref_phase, TC, order, show = False):
    '''
    Simulating measurement with Lock-in amplifier. Signal is multiplied by reference signal,
    filtered and returned. 

    Parameters
    ----------
    ts : np array
    sig : np array
    ref_freq : int
    ref_phase : int
    TC : integer
        time constant used in filter.
    order : integer 1-4
        determines the slope of the filter.
    show : boolean, optional
        plot the frequency response of the signal*reference before and after filtering. The default is False.

    Returns
    -------
    out_X : numpy array
        real component of the filtered signal.
    out_Y : numpy array
        imaginary component of the filtered signal.

    '''
    dt = ts[1]-ts[0]
    filt = lpf.rc_lpf(1.0/(2.0*np.pi*TC), order)
    
   
    ref_sig1 = np.cos((2.0 * np.pi * ref_freq * ts + np.pi/180.0*ref_phase))
    ref_sig2 = np.cos((2.0 * np.pi * ref_freq * ts + np.pi/180.0*(ref_phase+90.0)))
   
    p1f = np.fft.rfft(2.0 * np.real(sig) * ref_sig1)
    p2f = np.fft.rfft(2.0 * np.real(sig) * ref_sig2)
    fs = np.fft.rfftfreq(sig.size, dt)

  
    p1ff = filt(fs)*p1f
    p2ff = filt(fs)*p2f

    if show:
        plt.figure('unfiltered')
        plt.plot(fs,p1f, label = 'in phase', color = 'black')
        plt.plot(fs,p2f, label = 'out of phase', color = 'red')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
                
            
        plt.figure('filtered')
        plt.plot(fs,p1ff, label = 'in phase', color = 'black')
        plt.plot(fs,p2ff, label = 'out of phase', color = 'red')
        plt.xlabel('Frequency [Hz]')
        plt.legend()

    out_X = np.fft.irfft(p1ff, sig.size)
    out_Y = np.fft.irfft(p2ff, sig.size)


    return out_X, out_Y # returns signal amplitudes (not rms)



    
    