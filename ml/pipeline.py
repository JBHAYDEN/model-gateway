from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
NUMERIC=["age","sessions","purchases","average_session_minutes","purchase_per_session","engagement_score"]
CATEGORICAL=["channel"]
def build_pipeline()->Pipeline:
    numeric=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore"))])
    prep=ColumnTransformer([("numeric",numeric,NUMERIC),("categorical",categorical,CATEGORICAL)])
    return Pipeline([("preprocessing",prep),("classifier",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))])
