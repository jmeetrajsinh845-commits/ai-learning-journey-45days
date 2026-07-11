import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:


    def __init__(self):
        self.scaler = StandardScaler()



    def split_features_target(
        self,
        df,
        target
    ):

        X = df.drop(
            target,
            axis=1
        )

        y = df[target]


        return X,y



    def scale_features(
        self,
        X
    ):

        scaled_data = self.scaler.fit_transform(
            X
        )


        return pd.DataFrame(
            scaled_data,
            columns=X.columns
        )
