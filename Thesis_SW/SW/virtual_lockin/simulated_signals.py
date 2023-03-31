import numpy as np

def LHO_homo(IC, t0, f0, gamma):
  
    w0 = f0 * 2 * np.pi + 1.0j*gamma

    def cplx_pos(t = 0):
         return IC * np.exp(1.0j * w0 * (t-t0))
   
    def cplx_vel(t = 0):
        return IC * np.exp(1.0j * w0 * (t-t0)) * 1.0j*w0

    return cplx_pos, cplx_vel


def LHO_part(f0, drive, gamma, m):

    w0 = f0 * 2 * np.pi
  
    def cplx_pos(f, t = 0):
        w = f * 2 * np.pi
        denom = ((w0**2- w**2)**2 + w**2 * gamma**2)
        return drive/(m*denom) * ((w0**2-w**2)  -  1.0j* w * gamma) * np.exp(1.0j * w * t)
   
    def cplx_vel(f, t = 0):
        w = f * 2 * np.pi
        denom = ((w0**2- w**2)**2 + w**2 * gamma**2)
        return drive/(m*denom) * ((w0**2-w**2)  -  1.0j* w * gamma) * np.exp(1.0j * w * t) * 1.0j*w

    return cplx_pos, cplx_vel


def noisy_sin(ts, amp, freq, phase):
  
    w = freq*np.pi*2
    phase = phase*2*np.pi/360 #converting phase from degrees to radians 
    noise = np.random.normal(scale=0.05, size=len(ts))
    signal = amp * np.sin(w*ts+phase ) + noise
    
    return signal


def noisy_cos(ts, amp, freq, phase):

    w = freq*np.pi*2
    phase = phase*2*np.pi/360 #converting phase from degrees to radians 
    noise = np.random.normal(scale=0.05, size=len(ts))
    signal = amp * np.cos(w*ts+phase ) + noise
    
    return signal



