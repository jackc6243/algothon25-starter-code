import numpy as np
import pandas as pd


class BackTester:
    commissionFee = 0.00005
    maxPos = 10_000

    def __init__(self, nt, nInst, dataFileName):
        self.nt = nt
        self.nInst = nInst

        df = pd.read_csv(dataFileName, sep="\s+", header=None, index_col=None)
        (self.nt, self.nInst) = df.shape
        self.data = (df.values).T
        self.getPos = None

    def set_getPos(self, func):
        self.getPos = func

    def backtest(self):
        positions = np.zeros((self.nInst, self.nt))
        profits = np.zeros((self.nInst, self.nt))
        for t in range(0, self.nt):
            positions[:, t] = (
                self.getPos(self.data[:, :t]) if t < self.nt else positions[:, t - 1]
            )
            curPrices = self.data[:, t]
            prevPrices = self.data[:, t - 1] if t > 0 else np.zeros((self.nInst, 1))
            curPos = positions[:, t]
            prevPos = self.data[:, t - 1] if t > 0 else np.zeros((self.nInst, 1))

            profits[:, t] += (curPrices - prevPrices) * prevPos
            profits[:, t] -= self.commissionFee * abs((curPos - prevPos) * curPrices)

        return (positions, profits)
