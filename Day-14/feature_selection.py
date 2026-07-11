from sklearn.feature_selection import (
    mutual_info_classif,
    RFE
)

from sklearn.linear_model import LogisticRegression



class FeatureSelector:



    def mutual_information(
        self,
        X,
        y
    ):

        scores = mutual_info_classif(
            X,
            y
        )


        result = dict(
            zip(
                X.columns,
                scores
            )
        )


        return result



    def recursive_feature_elimination(
        self,
        X,
        y,
        number_features
    ):


        model = LogisticRegression(
            max_iter=1000
        )


        selector = RFE(
            model,
            n_features_to_select=
            number_features
        )


        selector.fit(
            X,
            y
        )


        selected = X.columns[
            selector.support_
        ]


        return list(selected)
