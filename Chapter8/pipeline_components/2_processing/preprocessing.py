from __future__ import absolute_import, division, print_function, unicode_literals
import click
import json
import os
import argparse
import dill
import pandas as pd
from sklearn.model_selection import train_test_split




@click.command()
@click.option('--data-file', default="/mnt/breast.data")
@click.option('--train-file', default="/mnt/training.data")
@click.option('--test-file', default="/mnt/test.data")
@click.option('--validation-file', default="/mnt/validation.data")
@click.option('--train-target', default="/mnt/trainingtarget.data")
@click.option('--test-target', default="/mnt/testtarget.data")
@click.option('--validation-target', default="/mnt/validationtarget.data")
@click.option('--split-size', default=0.1)
def training_data_processing(data_file,train_file,test_file,validation_file,split_size,train_target,test_target,validation_target):

    with open(data_file, 'rb') as in_f:
        data= dill.load(in_f)
        
    data=data.fillna(data.mean())


    target_name = 'target'
    data_target = data[target_name]
    data = data.drop([target_name], axis=1)
   
     #%% split training set to validation set
    train, test, target, target_test = train_test_split(data, data_target, test_size=split_size, random_state=0)
    Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=split_size, random_state=0)

    
    
    print(len(Xtrain), 'train examples')
    print(len(test), 'validation examples')
    print(len(Xval), 'test examples')
    


    '''
    Storage.upload(model_output_base_path,gcs_path)
    metadata = {
            'outputs' : [{
            'type': 'table',
            'storage': 'gcs',
            'format': 'csv',
            'header': [x['name'] for x in schema],
            'source': prediction_results
            }]
        }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
          json.dump(metadata, f)
    ''' 
    with open(train_file,"wb") as f:
        dill.dump(Xtrain,f) 
    
    with open(test_file,"wb") as f:
        dill.dump(test,f) 
        
    with open(validation_file,"wb") as f:
        dill.dump(Xval,f) 
    
    with open(train_target,"wb") as f:
        dill.dump(Ztrain,f) 
    
    with open(test_target,"wb") as f:
        dill.dump(target_test,f) 
        
    with open(validation_target,"wb") as f:
        dill.dump(Zval,f) 
    return


if __name__ == "__main__":
    training_data_processing()