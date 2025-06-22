import numpy as np
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPositionBase(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos + rpos])


def getMyPositionLinearRegression(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos + rpos])


def splitPricesData(original_file, test_days):
    df = pd.read_csv(original_file, sep=r"\s+", header=None, index_col=None)
    total_days, n_instruments = df.shape
    train_days = total_days - test_days
    if train_days <= 0:
        raise ValueError(
            f"test_days ({test_days}) cannot be greater than or equal to total days ({total_days})"
        )
    train_data = df.iloc[:train_days, :].values.T
    test_data = df.iloc[train_days:, :].values.T
    print(f"Split data: {train_days} days for training, {test_days} days for testing")
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    return train_data, test_data


def savePricesToCSV(data, filename):
    df = pd.DataFrame(data.T)
    df.to_csv(filename, sep="\t", header=False, index=False)
    print(f"Saved data to {filename}")


def splitAndSavePrices(
    original_file,
    test_days,
    train_filename="prices_train.txt",
    test_filename="prices_test.txt",
):
    train_data, test_data = splitPricesData(original_file, test_days)
    savePricesToCSV(train_data, train_filename)
    savePricesToCSV(test_data, test_filename)
    return train_data, test_data
