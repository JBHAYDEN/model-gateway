from ml.train import train_and_save
def test_training_creates_artifacts(tmp_path):
    metrics=train_and_save(tmp_path,7)
    assert (tmp_path/"conversion_model.joblib").exists()
    assert 0<=metrics["f1_score"]<=1
