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
    return currentPos


def getMyPositionLinearRegression(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    look_back_period = 7

    if nt < look_back_period + 1:
        return np.zeros(nins)

    positions = np.zeros(nins)

    for i in range(nins):
        # Prepare training data using sliding window
        X = []
        y = []
        # Create training examples: each window of look_back_period prices predicts the next price
        for t in range(nt - look_back_period):
            X.append(prcSoFar[i, t : t + look_back_period])
            y.append(prcSoFar[i, t + look_back_period])

        if len(X) == 0:  # Not enough data
            continue

        X = np.array(X)
        y = np.array(y)

        # Add bias term to X
        X_b = np.c_[np.ones(X.shape[0]), X]

        try:
            # Closed-form solution for linear regression
            w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

            # Use last look_back_period prices to predict next price
            x_pred = np.r_[1, prcSoFar[i, -look_back_period:]]
            next_price_pred = x_pred @ w

            # Calculate expected return
            current_price = prcSoFar[i, -1]
            expected_return = (next_price_pred - current_price) / current_price

            # Position sizing
            if abs(expected_return) > 0.001:  # Minimum threshold
                position_size = int(5000 * expected_return / current_price)
                positions[i] = position_size

        except np.linalg.LinAlgError:
            # Fallback to momentum if regression fails
            print("failed")
            if nt >= 2:
                momentum = (prcSoFar[i, -1] - prcSoFar[i, -2]) / prcSoFar[i, -2]
                if abs(momentum) > 0.001:
                    position_size = int(5000 * momentum / prcSoFar[i, -1])
                    positions[i] = position_size

    currentPos = positions
    return currentPos


def getMyPositionMeanReversion1(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    # Define constants
    SMA_PERIOD = 20
    EMA_PERIOD = 7

    if nt < SMA_PERIOD:
        return np.zeros(nins)

    # Initialize position array
    positions = np.zeros(nins)

    for i in range(nins):
        # Calculate 20-day SMA
        sma = np.mean(prcSoFar[i, -SMA_PERIOD:])

        # Calculate 7-day EMA
        prices = prcSoFar[i, -EMA_PERIOD:]
        ema = prices[0]  # Start with first price
        alpha = 2 / (EMA_PERIOD + 1)
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        # Generate signal based on crossover
        signal = 0
        if ema > sma:
            signal = 1  # Buy signal
        elif ema < sma:
            signal = -1  # Sell signal

        # Position sizing
        current_price = prcSoFar[i, -1]
        if current_price != 0:  # Avoid division by zero
            position_size = int(5000 * signal / current_price)
            positions[i] = position_size

    # Update global position
    currentPos = np.add(currentPos, positions).astype(int)
    return currentPos


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
