import numpy as np
import os

def AccessData():
    datasets = []
    data = []
    data_path = ''
    path = r"G:\My Drive\Diplomovka_folder\Data"     
    
    for path, subdirs, files in os.walk(path):
       for name in files:     
            if 'notes' in name:
                 note_name =  name
                 name = note_name.replace('_notes','')
                 notes_path = os.path.join(path, note_name)
                 data_path = os.path.join(path, name)
                 notes = np.loadtxt(notes_path, delimiter = ':', usecols = 1)
                 datasets = np.loadtxt(data_path, skiprows = 1)    
                 temp = (name, datasets, data_path, notes)
                 data.append(temp)
    return  data 

def SortData(datasets):
    # creating array of empty dictionaries
    dictlist = [dict() for ds in datasets]
 
    for count, ds in enumerate(datasets):
        name, data, path, note = ds
        values = []        
    
        #sorting datasets acording to external parameters
        for meas in type_of_measurement:
            if meas in path:
                values.append(meas)
                break
        
        for mag in magnetic_field_current:
            if mag in name:
                values.append(mag)
                break
        
        for temper in temperatures:
            if temper in path:
                values.append(temper)
                break
        
        for wire in wire_types:
           if wire in name:
               values.append(wire)
               break
          
        values.append(data)
        values.append(note[0])
        
        #populating dictionaries
        for i, key in enumerate(keys):       
            dictlist[count][key] = values[i] 
                         

    return dictlist
                       
def load_soarted_data():
    '''
    Returns
    -------
    list of dictionaries

    '''
    datasets = AccessData()
    dictlist = SortData(datasets)
    return dictlist
        
wire_types = ['G1L', 'G1R', 'G3R']  
temperatures = ['920', '700', '500', '250', 'baseT']  #mK
magnetic_field_current = ['0.1', '0.075', '0.5']        #A
type_of_measurement = ['Vacuum', 'Helium']                        
keys =  ['measurement', 'magnet_drive', 'temperature', 'wire','data', 'drive']



