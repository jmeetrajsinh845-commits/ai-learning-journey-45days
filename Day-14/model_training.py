from xgboost import XGBClassifier



class ModelTrainer:



    def train(
        self,
        X,
        y
    ):


        model = XGBClassifier(

            n_estimators=300,

            learning_rate=0.05,

            max_depth=4,

            random_state=42

        )


        model.fit(
            X,
            y
        )


        return model
