from pathlib import Path
import json,joblib
from sklearn.model_selection import GridSearchCV,train_test_split
from ml.dataset import generate_customer_data
from ml.features import add_behavior_features
from ml.pipeline import build_pipeline
from ml.evaluate import calculate_metrics

def train_and_save(artifact_dir:str|Path="artifacts",random_state:int=42)->dict[str,float]:
    frame=add_behavior_features(generate_customer_data(random_state=random_state))
    x=frame.drop(columns=["converted"]); y=frame["converted"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=random_state,stratify=y)
    search=GridSearchCV(build_pipeline(),{"classifier__C":[.1,1.0,10.0]},cv=5,scoring="f1",n_jobs=-1)
    search.fit(x_train,y_train)
    metrics=calculate_metrics(y_test,search.predict(x_test)); metrics["best_cv_f1"]=float(search.best_score_)
    out=Path(artifact_dir); out.mkdir(parents=True,exist_ok=True)
    joblib.dump(search.best_estimator_,out/"conversion_model.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics
