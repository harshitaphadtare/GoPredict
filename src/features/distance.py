import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

def calc_distance(df,method='manhattan'):

    '''Calculating Manhattan and Euclidean (Havesine) distances'''

    earthR = 6378137.0   # radius of Earth in meters
    pi180 = np.pi / 180  # conversion factor from degrees to radians

    #difference in latitude and longitude (in radians)
    dlat = (df.end_lat - df.start_lat) * pi180
    dlng = (df.end_lng - df.start_lng) * pi180

    if method == 'manhattan':
        #computing north-south distance between start and end lat assuming long fixed
        ay = np.sin(np.abs(dlat)/2)**2
        cy = 2* np.arctan2(np.sqrt(ay),np.sqrt(1-ay))
        dy = earthR * cy
        
        #computing east-west distance between start and end lng assuming lat fixed
        ax = np.sin(np.abs(dlng)/2)**2
        cx = 2* np.arctan2(np.sqrt(ax),np.sqrt(1-ax))
        dx = earthR * cx

        distance = np.abs(dx) + np.abs(dy)
    
    elif method == "euclidean":
        a = (np.sin(dlat/2)**2 + np.cos(df.start_lat*pi180) * np.cos(df.end_lat*pi180) * np.sin(dlng/2)**2)
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance = earthR * c
    
    else:
        distance = 0
    
    return distance
