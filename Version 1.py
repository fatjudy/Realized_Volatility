import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import glob

def train_test(mode):
    # mode = "train"/"test"
    file_name =  mode + '.csv'
    return pd.read_csv(file_name)

train = train_test("train")
train.head()

order_book_training = glob.glob('book_train.parquet/*')

# The two function below comes from: https://www.kaggle.com/shahmahdihasan/overly-simplified-ols-prediction
# custom aggregate function
def wap2vol(df):
    # wap2vol stands for WAP to Realized Volatility
    temp = np.log(df).diff()  # calculating tik to tik returns
    # returning realized volatility
    return np.sqrt(np.sum(temp ** 2))


# function for calculating realized volatility per time id for a given stock
def rel_vol_time_id(path):
    # book: book is an order book
    book = pd.read_parquet(path)  # order book for a stock id loaded
    # calculating WAP
    p1 = book["bid_price1"]
    p2 = book["ask_price1"]
    s1 = book["bid_size1"]
    s2 = book["ask_size1"]

    book["WAP"] = (p1 * s2 + p2 * s1) / (s1 + s2)
    # calculating realized volatility for each time_id
    # transbook = book.groupby("time_id")["WAP"].agg(wap2vol)

    return book[['time_id', 'seconds_in_bucket', 'WAP']]

rel_vol_time_id(order_book_training[0])

stock_id = []
time_id = []
relvol = []
seconds=  []
for i in order_book_training:
    # finding the stock_id
    temp_stock = int(i.split("=")[1])
    # find the realized volatility for all time_id of temp_stock
    temp_relvol = rel_vol_time_id(i)
    stock_id += [temp_stock]*temp_relvol.shape[0]
    seconds += list(temp_relvol['seconds_in_bucket'])
    time_id += list(temp_relvol.index)
    relvol += list(temp_relvol['WAP'])

past_volatility = pd.DataFrame({"stock_id": stock_id, "time_id": time_id,"seconds": seconds,
                                "realized_volatility": relvol})

joined = train.merge(past_volatility, on=["stock_id", "time_id"], how="left")
joined.head()

