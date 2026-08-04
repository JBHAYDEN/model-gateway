import pandas as pd
from ml.features import add_behavior_features
def test_features():
    result=add_behavior_features(pd.DataFrame({"sessions":[4.0],"purchases":[2.0],"average_session_minutes":[10.0]}))
    assert result.loc[0,"purchase_per_session"]==.5
    assert result.loc[0,"engagement_score"]==40.0
