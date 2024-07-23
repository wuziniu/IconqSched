import numpy as np
from xgboost import XGBRegressor


class SingleXGBoost:
    def __init__(
        self,
        n_estimators=1000,
        max_depth=8,
        eta=0.2,
        eval_metric="mae",
        early_stopping_rounds=100,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            eta=eta,
            subsample=1.0,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
        )

    def train(self, df):
        features = df["features"].values
        features = np.asarray(list(features))
        labels = df["runtime"].values
        train_idx = np.random.choice(
            len(features), size=int(0.8 * len(features)), replace=False
        )
        val_idx = [i for i in range(len(features)) if i not in train_idx]
        self.model.fit(
            features[train_idx],
            labels[train_idx],
            eval_set=[(features[val_idx], labels[val_idx])],
            verbose=False,
        )

    def predict(self, df):
        features = df["features"].values
        if len(features.shape) == 1 and isinstance(features[0], np.ndarray):
            features = np.stack(features)
        preds = self.model.predict(features)
        preds = np.maximum(preds, 0.01)
        return preds

    def online_inference(self, feature: np.ndarray) -> float:
        feature = feature.reshape(1, -1)
        pred = self.model.predict(feature)[0]
        return pred
