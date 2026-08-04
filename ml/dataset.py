from __future__ import annotations
import numpy as np
import pandas as pd

def generate_customer_data(rows:int=800, random_state:int=42)->pd.DataFrame:
    rng=np.random.default_rng(random_state)
    age=rng.integers(18,70,size=rows)
    sessions=rng.poisson(6,size=rows)
    purchases=rng.poisson(1.5,size=rows)
    minutes=rng.gamma(2.5,4.0,size=rows)
    channel=rng.choice(["organic","search","social","referral"],size=rows,p=[.35,.30,.20,.15])
    score=.04*age+.20*sessions+.75*purchases+.05*minutes+np.where(channel=="referral",.8,0)-5.4
    converted=rng.binomial(1,1/(1+np.exp(-score)))
    frame=pd.DataFrame({"age":age.astype(float),"sessions":sessions.astype(float),"purchases":purchases.astype(float),
      "average_session_minutes":minutes,"channel":channel,"converted":converted})
    idx=rng.choice(frame.index,size=max(1,rows//25),replace=False)
    frame.loc[idx,"average_session_minutes"]=np.nan
    return frame
