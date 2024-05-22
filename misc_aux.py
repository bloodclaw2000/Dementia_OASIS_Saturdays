import pandas as pd 

def getDataFrameFromDict (dict):
              properties = ['ID', 'M/F', 'Hand', 'Age', 'Educ', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'Delay', 'Dementia']
              subset = [[value['1'][prop] for prop in properties] for key, value in dict.items()]
              return pd.DataFrame(subset,columns=properties)