import pandas as pd


def check_data_change(
        old_data,
        new_data
):


    old_mean=old_data.mean()

    new_mean=new_data.mean()



    difference=abs(
        old_mean-new_mean
    )



    print(
        "Data Change Report"
    )


    print(
        difference
    )



    if difference.max()>10:

        print(
        "Warning: Data drift detected"
        )


    else:

        print(
        "Data looks stable"
        )



old=pd.read_csv(
    "../data/churn_data.csv"
)


new=pd.read_csv(
    "../data/churn_data.csv"
)


check_data_change(
    old,
    new
)
