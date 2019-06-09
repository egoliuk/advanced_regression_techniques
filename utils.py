import time
import os
import pandas as pd

def save_submission(ids, prices):
    submission=pd.DataFrame()
    submission['Id']=ids
    submission['SalePrice']=prices
    
    dirPath = "submissions"
    os.makedirs(dirPath, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filePath = f"{dirPath}/submission-{timestr}.csv"

    submission.to_csv(filePath, index=False)
    return filePath
