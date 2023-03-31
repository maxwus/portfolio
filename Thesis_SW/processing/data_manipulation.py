import numpy as np

def SubtractBackground(datasets):
    
    first_points = datasets[1:5]
    background = np.mean(first_points) 
    corrected_data = [d - background for d in datasets]
        
    return corrected_data
 

   
def FindAmplitude(dataset):

    minimum = (np.mean(dataset[:3]) + np.mean(dataset[-3:]))/2
    maximum = np.amax(dataset)
    amplitude = maximum - minimum
    
    return amplitude


def FindAmplitudeNoBgnd(dataset):

    amplitude = np.amax(dataset)
    
    return amplitude



def ElCurrent(Drive, Resistance, attenuation):

    I = np.sqrt(2) * Drive / Resistance * attenuation

    return I


def MagField(El_current):
    
    MagField = El_current * 0.126  # 1A = 0.126 T
    
    return MagField



def DrivingForce(magnetic_field, wire_length, electrical_current):

    F = magnetic_field * wire_length * electrical_current
    
    return F    

def WireVelocity(voltage, magnetic_field, wire_length):
    
    velocity = np.sqrt(2) * voltage/magnetic_field/wire_length 
    
    return velocity





