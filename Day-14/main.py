from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.model_training import ModelTrainer



loader = DataLoader(
    "data/churn_data.csv"
)


df = loader.load_data()



processor = DataPreprocessor()


X,y = processor.split_features_target(
    df,
    "churn"
)



selector = FeatureSelector()


features = selector.recursive_feature_elimination(
    X,
    y,
    5
)



print(
    "Selected Features:"
)

print(features)



X_selected = X[
    features
]



trainer = ModelTrainer()


model = trainer.train(
    X_selected,
    y
)


print(
    "Model Training Completed"
)
