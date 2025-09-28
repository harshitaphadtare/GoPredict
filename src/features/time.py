import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import datetime


def extract_time_features(combine_df):
    '''Extract weekdays,hour and date columns and drop datetime
    column. Add holidays column'''

    holidays2015 = {
        "New Years Day": datetime.datetime(2015,1,1).date(),
        "Martin Luther King Day": datetime.datetime(2015,1,19).date(),
        "Easter Saturday": datetime.datetime(2015,4,4).date(),
        "Easter Sunday": datetime.datetime(2015,4,5).date(),
        "Memorial Sunday": datetime.datetime(2015,5,24).date(),
        "Memorial Day": datetime.datetime(2015,5,25).date(),
        "Independence Pre-day": datetime.datetime(2015,7,3).date(),
        "Independence Day": datetime.datetime(2015,7,4).date(),
        "Independence Post-day": datetime.datetime(2015,7,5).date(),
        "Labor Day": datetime.datetime(2015,9,7).date(),
        "Thanksgiving Day": datetime.datetime(2015,11,26).date(),
        "Thanksgiving Post-day": datetime.datetime(2015,11,27).date(),
        "Thanksgiving Post-post-day": datetime.datetime(2015,11,28).date(),
        "Christmas Eve": datetime.datetime(2015,12,24).date(),
        "Christmas Day": datetime.datetime(2015,12,25).date(),
        "Christmas Post-day": datetime.datetime(2015,12,26).date(),
        "New Years Eve": datetime.datetime(2015,12,31).date(),
    }

    for df in combine_df:
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        df['weekday'] = df['datetime'].dt.weekday + 1
        df['hour'] = df['datetime'].dt.hour
        df['date'] = df['datetime'].dt.date

        df.drop(columns=['datetime'], inplace=True)

        df['holiday'] = 0
        df.loc[df['date'].isin(holidays2015.values()), 'holiday'] = 1
    
    return combine_df
