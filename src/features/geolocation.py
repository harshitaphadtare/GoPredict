import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import gmplot
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def prepare_coordinates(train_df,test_df):
    '''Combines train/test into one coordinate dataframe 
    with start/end labels'''

    train_df['train_or_test'] = 'train'
    test_df['train_or_test'] = 'test'
    train_test = pd.concat([train_df,test_df]).reset_index()

    start = pd.DataFrame(
        train_test[['row_id','start_lat','start_lng','train_or_test']].values,
        columns=['row_id','lat','lng','train_or_test']
    )

    start['start_or_end'] = 'start'

    end = pd.DataFrame(
        train_test[['row_id','end_lat','end_lng','train_or_test']].values,
        columns=['row_id','lat','lng','train_or_test']
    )

    end['start_or_end'] = 'end'

    coordinates = pd.concat([start,end]).reset_index(drop=True)
    return coordinates


def cluster_coordinates(coordinates,eps=0.005,min_sample=2500):
    '''Run DBSCAN clustering on coordinates to group locations 
    into clusters of airports,city centers etc'''

    db = DBSCAN(eps=eps,min_samples=min_sample).fit(coordinates[['lat','lng']])
    print("Number of clusters found:",max(db.labels_)+1)
    return db


def label_clusters(db):
    '''Assign readable cluster labels.'''

    labels = []
    for i in db.labels_:
        if i == -1:
            labels.append('standalone')
        elif i in [0,1,3,4]:
            labels.append('city')
        elif i in [2,5,6]:
            labels.append('airport')
    return np.array(labels)


def visualize_clusters(coordinates,db):
    '''Generate heatmaps for each cluster using gmplot'''
    try:
        for i in range(max(db.labels_)+1):
            cluster = coordinates.loc[np.argwhere(db.labels_ == i).flatten()]
            gmap = gmplot.GoogleMapPlotter(
                cluster.lat.sample(1).values[0],
                cluster.lng.sample(1).values[0],
                12
            )
            gmap.heatmap(cluster.lat,cluster.lng)
            gmap.draw(f'gmaps/cluster{i}.html')
    except Exception as e:
        logging.warning(f"Could not generate cluster visualizations: {e}")


def add_cluster_features(train_df,test_df,coordinates,labels):
    ''' Map cluster features back to train/test dataframes'''

    coordinates['location'] = labels

    train_locs = coordinates.loc[coordinates.train_or_test == 'train']
    test_locs = coordinates.loc[coordinates.train_or_test == 'test']

    train_start = train_locs[train_locs.start_or_end == 'start'][['row_id','location']].set_index('row_id')
    train_end = train_locs[train_locs.start_or_end == 'end'][['row_id','location']].set_index('row_id')
    test_start = test_locs[test_locs.start_or_end == 'start'][['row_id','location']].set_index('row_id')
    test_end = test_locs[test_locs.start_or_end == 'end'][['row_id','location']].set_index('row_id')

    train_df['start_loc'] = train_start
    train_df['end_loc'] = train_end
    test_df['start_loc'] = test_start
    test_df['end_loc'] = test_end

    for df in [train_df,test_df]:
        df['airport'] = ((df.start_loc == 'airport') | (df.end_loc == 'airport')).astype(int)
        df['citycenter'] = ((df.start_loc == 'city') | (df.end_loc == 'city')).astype(int)
        df['standalone'] = ((df.start_loc == 'standalone') | (df.end_loc == 'standalone')).astype(int)
        df.drop(columns=['start_loc','end_loc','train_or_test'], axis=1, inplace=True)

    return [train_df,test_df]


def clustering(train_df,test_df):
    '''final clustering function'''

    logging.info("Preparing coordinates...")
    coordinates = prepare_coordinates(train_df,test_df)

    logging.info("Preparing Clusters...")
    db = cluster_coordinates(coordinates)

    logging.info("Preparing labels for cluster...")
    labels = label_clusters(db)

    logging.info("Adding cluster features to dataframes...")
    add_cluster_features(train_df,test_df,coordinates,labels)

    logging.info("Saved HTML Files in gmaps/...")
    