import pandas as pd
import numpy as np

def get_most_volatile(prices):
    """Return the ticker symbol for the most volatile stock.
    
    Parameters
    ----------
    prices : pandas.DataFrame
        a pandas.DataFrame object with columns: ['ticker', 'date', 'price']
    
    Returns
    -------
    ticker : string
        ticker symbol for the most volatile stock
    """
    # TODO: Fill in this function.
    prices_v = prices.copy()

    # print(prices_v)
    # print()
    # print("raw_return")
    prices_v["price"] = (prices_v["price"] - prices_v["price"].shift(1)) / prices_v["price"].shift(1) #raw_return
    # print(prices_v)

    # print()
    # print("log_return")
    prices_v["price"] = np.log(prices_v["price"] + 1) #log_reurn


    prices_group = prices_v.groupby("ticker").std()

    # print()
    # print("test std")
    # print(prices_group)
    # print(f"std.type: {type(prices_group)}")

    #max = prices_group[prices_group["price"].max()]
    max = prices_group.loc[prices_group['price'].idxmax()]
    # print()
    # print(f"max:")
    # print(max)
    # print(f"max.type: {type(max)}")
    # print(f"max.name: {max.name}")
    # print(f"max.index: {max.index}")

    return max.name


def test_run(filename='./data/prices.csv'):
    """Test run get_most_volatile() with stock prices from a file."""
    prices = pd.read_csv(filename, parse_dates=['date'])
    print("Most volatile stock: {}".format(get_most_volatile(prices)))


if __name__ == '__main__':
    test_run()
