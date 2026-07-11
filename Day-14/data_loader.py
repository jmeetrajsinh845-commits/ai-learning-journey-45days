import pandas as pd


class DataLoader:

    def __init__(self, filepath):
        self.filepath = filepath


    def load_data(self):

        try:
            df = pd.read_csv(self.filepath)

            print(
                f"Dataset Loaded Successfully"
            )

            print(
                f"Rows: {df.shape[0]}"
            )

            print(
                f"Columns: {df.shape[1]}"
            )

            return df


        except Exception as error:

            print(
                "Data loading failed:",
                error
            )

            return None
