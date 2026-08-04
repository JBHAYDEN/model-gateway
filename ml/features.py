import pandas as pd

def add_behavior_features(frame:pd.DataFrame)->pd.DataFrame:
    result=frame.copy()
    safe=result["sessions"].replace(0,1)
    result["purchase_per_session"]=result["purchases"]/safe
    result["engagement_score"]=result["sessions"]*result["average_session_minutes"].fillna(0)
    return result
