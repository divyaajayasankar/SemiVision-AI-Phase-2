import pandas as pd
from src.data import leakage_report

def test_cross_split_duplicate_detection():
    df=pd.DataFrame({'filename':['a','b','c'],'split':['train','validation','test']}); assert leakage_report(df)['passed']
    bad=pd.DataFrame({'filename':['a','a'],'split':['train','test']}); assert not leakage_report(bad)['passed']
