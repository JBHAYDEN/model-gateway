from ml.train import train_and_save
if __name__=="__main__":
    metrics=train_and_save()
    for name,value in metrics.items(): print(f"{name}: {value:.4f}")
