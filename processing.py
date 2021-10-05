# from drawSvg.elements import DrawingBasicElement

# https://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas

import pymongo
import pandas as pd
import numpy as np
import math
import os
from pathlib import Path
from pymongo import MongoClient
from PIL import Image

def build_data(rebuild=False, img_size=512, limit_records=None):
    pickle_file = 'dataset.pickle'
    group_ids_file = 'ids.txt'
    result_ids = []

    # We will have to recreate pickle from db if it is not exist
    if not os.path.exists(pickle_file):
        rebuild = True

    if rebuild:
        client = MongoClient('mongodb://192.168.1.49:27017')
        db = client.geometry3
        objects = db.objects

        group_ids = list(objects.find().distinct('GroupId'))
        df = pd.DataFrame()

        for i, group_id in enumerate(group_ids):
            if limit_records and i > limit_records:
                break
            data = query_collection_to_dataframe(db, group_id, img_size)

            if data is not None:
                df = pd.concat([df, data])
                result_ids.append(group_id)

        df['ClassName'] = df['ClassName'].astype('category')
        df['GroupId'] = df['GroupId'].astype('category')

        df.to_pickle(pickle_file)
        with open(group_ids_file, 'w') as f:
            for id in result_ids:
                f.write(id+'\n')

    else:
        df = pd.read_pickle(pickle_file)
        with open(group_ids_file) as f:
            result_ids = f.read().splitlines()

    return df, result_ids

def query_collection_to_dataframe(db=None, group_id=None, img_size=512, max_entities=250, min_entities=8):
    '''
    Queries mongo objects collection to dataframe.
    Expands certain columns, like StartPoint to StartPoint.X, StartPoint.Y
    Scales each sample
    Returns pandas dataframe with given columns.
    If db has more entries than max_entrities, or less tha min_entities, returns None
    If problems, returns None

    If db is None, connects new client to collection

    If group_id is None returns single predefined drawing for test purposes
    '''

    source_images_dir = Path("data/images")

    # If collection is not specified
    if db is None:
        client = MongoClient('mongodb://192.168.0.102:27017')
        db = client.geometry3
    objects = db.objects
    fragments = db.fragments

    # Just arbitrary drawing
    if group_id is None:
        group_id = '1317d221-8d9e-4e2e-b290-3be2a0aa67fb'

    # first we query mongo collection for lines, texts and dimensions
    query = {'GroupId' : group_id}
    all_entities = list(objects.find(query))

    # filter out tables and schemes that doesn't contain annotations
    not_text_line = {'GroupId':group_id, 'ClassName':{'$nin':['Line', 'Text', 'Entity', 'Hatch', 'Polyline','AcDbBlockReference','Circle']}}
    if objects.count(not_text_line) == 0:
        return

    # check number of entities per fragment
    if (len(all_entities) < min_entities or len(all_entities) > max_entities):
        return

    drawing_annotation_classes = ['AlignedDimension']#, 'McNotePosition',  'LevelMark', 'WeldDesignation', 'Axis', 'Section', 'McNoteChain', 'ViewDesignation']
    query_annotations = {
        'GroupId' : group_id,
        'ClassName': {'$in':drawing_annotation_classes}
    }
    drawing_annotations = list(objects.find(query_annotations))

    #check group_id contain annotations
    #if len(drawing_annotations) == 0:
    #    return

    # now we create dataframe
    df = pd.DataFrame(drawing_annotations)

    # We expand object point columns to point coordinates
    cols_to_expand = ['XLine1Point', 'XLine2Point', 'DimLinePoint']
    description_cols = ['GroupId', 'ClassName', 'StrippedFileName', 'AnnotatedFileName']
    df = expand_columns(df, cols_to_expand)

    # and return only dataframe with given columns
    dataframe_cols = []
    for col in cols_to_expand:
        dataframe_cols.append(col+'.X')
        dataframe_cols.append(col+'.Y')
    dataframe_cols += description_cols

    # fill in empty df columns
    for col in dataframe_cols:
        if col not in df.columns:
            df[col] = np.nan

    fragment = fragments.find_one({'GroupId':group_id})
    if fragment is not None:
        # strppd_file = Path(fragment['StrippedFileName'])
        # anntd_file = Path(fragment['AnnotatedFileName'])
        stripped_file = source_images_dir / ('stripped_' + group_id + '.png')
        annotated_file = source_images_dir / ('annotated_' + group_id + '.png')
        if stripped_file.exists:
            df['StrippedFileName'] = str(stripped_file)
        if annotated_file.exists:
            df['AnnotatedFileName'] = str(annotated_file)
        bound_min_x = fragment['MinBoundPoint']['X']
        bound_min_y = fragment['MinBoundPoint']['Y']
        bound_max_x = fragment['MaxBoundPoint']['X']
        bound_max_y = fragment['MaxBoundPoint']['Y']
        # we normalize dataframe
        df = normalize(df=df, to_size=img_size, base_pnt_x=bound_min_x, base_pnt_y=bound_min_y, diff_x=bound_max_x - bound_min_x, diff_y=bound_max_y - bound_min_y)

        print('group_id: {} df len: {}'.format(group_id, len(df)))

        return df[dataframe_cols]

def normalize(df, to_size=512, base_pnt_x=0, base_pnt_y=0, diff_x=None, diff_y=None):
    '''
    scales coordinate dataframe columns containig .X or .Y
    to be in [0...to_size] range
    Coordinates are shifted in accordance to fragment's base_pnt_x, base_pnt_y
    '''
    xcols = []
    ycols = []
    for column in df.columns:
        if ".X" in column:
            xcols.append(column)
        if ".Y" in column:
            ycols.append(column)

    # keep in mind that coordinates are stored in drawing coordinates
    # while we need to shift them to fragment's coordinates
    df[xcols] -= base_pnt_x
    df[ycols] -= base_pnt_y

    cols = xcols + ycols
    coords = df[cols]

    assert diff_x > 0 and diff_y > 0,  "Bound diffs should be positive"

    diff = max(diff_x, diff_y)

    # https://stackoverflow.com/questions/38134012/pandas-dataframe-fillna-only-some-columns-in-place
    #coords[xcols] = coords[xcols].fillna(min_coord_x)
    #coords[ycols] = coords[ycols].fillna(min_coord_y)

    # https://stackoverflow.com/questions/44471801/zero-size-array-to-reduction-operation-maximum-which-has-no-identity
    if (not np.any(coords.to_numpy())):
        return df
    
    # print(coords)
    scale = to_size/diff

    # print(min_coord, scale)
    df[xcols] = (coords[xcols]) * scale
    df[ycols] = (coords[ycols]) * scale

    return  df
    
# https://stackoverflow.com/questions/49081097/slice-pandas-dataframe-json-column-into-columns
def expand_columns(df, column_names):
    '''
    expands dataframe df columns from column_names list
    that contain point objects, i.e.
    StartPoint column containing {'ClassName':.., "X":.., "Y":.. ...}
    will become several columns StartPoint.ClassName, StartPoint.X, StartPoint.Y ... etc. with 
    respected values
    '''
    res = df
    for col_name in column_names:
        
        if col_name not in df.columns:
            continue
        
        # get rid of nans in column, so the type can be determined and parsed
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html 
        res1 = df.dropna(subset = [col_name])
        values = res1[col_name].values.tolist()
        indexes = res1[col_name].index
        # print(values, indexes)
        # as we dropped some rows to get rid of nans
        # we need to keep index so rows can be matched between list 
        # and source dataset
        res1 = pd.DataFrame(data= values, index = indexes)
        #print(res1)
        res1.columns = col_name +"."+ res1.columns

        # res = res.drop(col_name, axis = 1)
        
        # Keep index!!
        res = pd.concat([res, res1], axis = 1)
    return res

if __name__ == "__main__":
    df,_ = build_data(rebuild=True, img_size=64, limit_records=100)