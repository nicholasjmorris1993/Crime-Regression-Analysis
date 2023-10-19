def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, KBinsDiscretizer, PolynomialFeatures
from xgboost.sklearn import XGBRegressor
import keras
from keras import layers, optimizers, regularizers
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kstest
from itertools import combinations
import pandas_datareader as pdr
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

if os.name == "nt":
    path_sep = "\\"
else:
    path_sep = "/"


class Regression:
    def __init__(
        self, 
        name="Regression Analysis", 
        path=None, 
        deep=True,
        rename=True, 
        time=True, 
        binary=True, 
        imputation=True, 
        variance=True,
        scale=True,
        atwood=True,
        binning=True,
        reciprocal=True, 
        interaction=True, 
        selection=True,
        tune=True,
        plots=True,
    ):
        self.name = name  # name of the analysis
        self.path = path  # the path where results will be exported
        self.deep = deep  # should we build a deep neural network (10 layers) or a shallow one (2 layers)?
        self.rename = rename  # should features be renamed to remove whitespace?
        self.time = time  # should datetime features be computed?
        self.binary = binary  # should categorical features be converted to binary features?
        self.imputation = imputation  # should missing values be filled in?
        self.variance = variance  # should we remove constant features?
        self.scale = scale  # should we scale the features?
        self.atwood = atwood  # should we compute atwood numbers?
        self.binning = binning  # should we put continous features into bins?
        self.reciprocal = reciprocal  # should reciporcals be computed?
        self.interaction = interaction  # should interactions be computed?
        self.selection = selection  # should we perform feature selection?
        self.tune = tune  # should we tune the model?
        self.plots = plots  # should we plot the analysis?

        if self.path is None:
            self.path = os.getcwd()

        # create folders for output files
        self.folder(f"{self.path}{path_sep}{self.name}")
        self.folder(f"{self.path}{path_sep}{self.name}{path_sep}dump")  # machine learning pipeline and data
        if self.plots:
            self.folder(f"{self.path}{path_sep}{self.name}{path_sep}plots")  # html figures

    def validate(self, X, y):
        # raw data
        self.X = X.copy()
        self.y = y.copy()

        # split up the data into training and testing
        trainX = X.head(int(0.8 * X.shape[0]))
        trainy = y.head(int(0.8 * y.shape[0]))
        testX = X.tail(int(0.2 * X.shape[0])).reset_index(drop=True)
        testy = y.tail(int(0.2 * y.shape[0])).reset_index(drop=True)

        print("Model Training:")
        start = time.time()

        # set up the machine learning pipeline
        self.names1 = FeatureNames(self.rename)
        self.datetime = TimeFeatures(self.time)
        self.categorical = CategoricalFeatures(self.binary)
        self.names2 = FeatureNames(self.rename)
        self.impute = ImputeFeatures(self.imputation)
        self.constant1 = ConstantFeatures(self.variance)
        self.selection1 = FeatureSelector(self.selection)
        self.scaler1 = ScaleFeatures(self.atwood or self.reciprocal, bounds=(0.2, 0.8))
        self.numbers = AtwoodNumbers(self.atwood)
        self.bin = BinFeatures(self.binning)
        self.reciprocals = Reciprocals(self.reciprocal)
        self.interactions = Interactions(self.interaction)
        self.constant2 = ConstantFeatures(self.reciprocal and self.interaction)
        self.selection2 = FeatureSelector(self.selection and (self.atwood or self.binning or self.reciprocal or self.interaction))
        self.scaler2 = ScaleFeatures(self.scale, bounds=(0, 1))
        
        # run the pipeline on training data
        print("> Transforming The Training Data")
        trainX = self.names1.fit_transform(trainX)
        trainX = self.datetime.fit_transform(trainX)
        trainX = self.categorical.fit_transform(trainX)
        trainX = self.names2.fit_transform(trainX)
        trainX = self.impute.fit_transform(trainX)
        trainX = self.constant1.fit_transform(trainX)
        trainX = self.selection1.fit_transform(trainX, trainy)
        trainX = self.scaler1.fit_transform(trainX)
        numbers = self.numbers.fit_transform(trainX)
        trainX = self.bin.fit_transform(trainX)
        trainX = self.reciprocals.fit_transform(trainX)
        trainX = self.interactions.fit_transform(trainX)
        trainX = pd.concat([trainX, numbers], axis="columns")
        trainX = self.constant2.fit_transform(trainX)
        trainX = self.selection2.fit_transform(trainX, trainy)
        trainX = self.scaler2.fit_transform(trainX)
        self.grid(trainX, trainy)

        end = time.time()
        self.run_time(start, end)

        print("Model Performance:")
        start = time.time()

        # transform the testing data and score the performance
        print("> Transforming The Testing Data")
        testX = self.names1.transform(testX)
        testX = self.datetime.transform(testX)
        testX = self.categorical.transform(testX)
        testX = self.names2.transform(testX)
        testX = self.impute.transform(testX)
        testX = self.constant1.transform(testX)
        testX = self.selection1.transform(testX)
        testX = self.scaler1.transform(testX)
        numbers = self.numbers.transform(testX)
        testX = self.bin.transform(testX)
        testX = self.reciprocals.transform(testX)
        testX = self.interactions.transform(testX)
        testX = pd.concat([testX, numbers], axis="columns")
        testX = self.constant2.transform(testX)
        testX = self.selection2.transform(testX)
        testX = self.scaler2.transform(testX)
        print("> Scoring The Model")
        self.performance(testX, testy)

        end = time.time()
        self.run_time(start, end)

        print("Model Indicators:")
        start = time.time()
        
        print("> Perturbing Features")
        self.importance(trainX, trainy)

        end = time.time()
        self.run_time(start, end)

    def fit(self, X, y):
        # raw data
        self.X = X.copy()
        self.y = y.copy()

        print("Model Training:")
        start = time.time()

        # set up the machine learning pipeline
        self.names1 = FeatureNames(self.rename)
        self.datetime = TimeFeatures(self.time)
        self.categorical = CategoricalFeatures(self.binary)
        self.names2 = FeatureNames(self.rename)
        self.impute = ImputeFeatures(self.imputation)
        self.constant1 = ConstantFeatures(self.variance)
        self.selection1 = FeatureSelector(self.selection)
        self.scaler1 = ScaleFeatures(self.atwood or self.reciprocal, bounds=(0.2, 0.8))
        self.numbers = AtwoodNumbers(self.atwood)
        self.bin = BinFeatures(self.binning)
        self.reciprocals = Reciprocals(self.reciprocal)
        self.interactions = Interactions(self.interaction)
        self.constant2 = ConstantFeatures(self.reciprocal and self.interaction)
        self.selection2 = FeatureSelector(self.selection and (self.atwood or self.binning or self.reciprocal or self.interaction))
        self.scaler2 = ScaleFeatures(self.scale, bounds=(0, 1))
        
        # run the pipeline on the data
        print("> Transforming The Data")
        X = self.names1.fit_transform(X)
        X = self.datetime.fit_transform(X)
        X = self.categorical.fit_transform(X)
        X = self.names2.fit_transform(X)
        X = self.impute.fit_transform(X)
        X = self.constant1.fit_transform(X)
        X = self.selection1.fit_transform(X, y)
        X = self.scaler1.fit_transform(X)
        numbers = self.numbers.fit_transform(X)
        X = self.bin.fit_transform(X)
        X = self.reciprocals.fit_transform(X)
        X = self.interactions.fit_transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.fit_transform(X)
        X = self.selection2.fit_transform(X, y)
        X = self.scaler2.fit_transform(X)
        self.grid(X, y)

        end = time.time()
        self.run_time(start, end)

        print("Model Indicators:")
        start = time.time()
        
        print("> Perturbing Features")
        self.importance(X, y)

        end = time.time()
        self.run_time(start, end)

    def grid(self, X, y):
        if self.tune:
            print("> Tuning Neural Network")
            print("> Cross Validating 15 Models")
            # build a grid search
            learning_rate = [0.0001, 0.001, 0.01]
            nodes = [32, 64, 128, 256, 512]
            fold = [0, 1, 2]
            grid = np.array(np.meshgrid(
                learning_rate, 
                nodes, 
                fold,
            )).reshape(3, len(learning_rate) * len(nodes) * len(fold)).T
            grid = pd.DataFrame(grid, columns=["learning_rate", "nodes", "fold"])
            grid["model"] = np.repeat(np.arange(15), 3)
            grid["fold"] = grid["fold"].astype(int)
            grid["nodes"] = grid["nodes"].astype(int)

            # generate the folds
            np.random.seed(0)
            folds = np.random.randint(low=0, high=3, size=X.shape[0])

            # score each model in the grid
            rmse = list()
            for i in range(grid.shape[0]):
                # split the data into training and testing
                test = np.where(folds == grid["fold"][i])[0]
                train = np.where(folds != grid["fold"][i])[0]
                trainX = X.copy().iloc[train, :].reset_index(drop=True)
                trainy = y.copy().iloc[train, :].reset_index(drop=True)
                testX = X.copy().iloc[test, :].reset_index(drop=True)
                testy = y.copy().iloc[test, :].reset_index(drop=True)

                # train the neural network
                if self.deep:
                    width = 10
                else:
                    width = 2
                self.nnet = self.build(
                    inputs=trainX.shape[1], 
                    outputs=trainy.shape[1], 
                    hidden=np.repeat(grid["nodes"][i], width), 
                    learning_rate=grid["learning_rate"][i], 
                    l1_penalty=0,
                )
                self.nnet.fit(trainX, trainy, epochs=50, batch_size=16, workers=-1, verbose=0)

                # score the neural network
                predictions = self.nnet.predict(testX, verbose=0).ravel()
                rmse.append(mean_squared_error(
                    y_true=testy.iloc[:, 0].to_numpy(),
                    y_pred=predictions,
                    squared=False,
                ))

            # get the best model
            grid["rmse"] = rmse
            grid = grid.groupby("model").agg({
                "learning_rate": "mean", 
                "nodes": "mean",
                "rmse": "mean",
            })
            grid = grid.sort_values(by="rmse", ascending=True).reset_index(drop=True)
            learning_rate = grid["learning_rate"][0]
            nodes = int(grid["nodes"][0])

            print("> Training The Best Model")
            if self.deep:
                width = 10
            else:
                width = 2
            self.nnet = self.build(
                inputs=X.shape[1], 
                outputs=y.shape[1], 
                hidden=np.repeat(nodes, width), 
                learning_rate=learning_rate, 
                l1_penalty=0,
            )
            self.nnet.fit(X, y, epochs=500, batch_size=16, workers=-1, verbose=0)
        else:
            print("> Training Neural Network")
            if self.deep:
                width = 10
            else:
                width = 2
            self.nnet = self.build(
                inputs=X.shape[1], 
                outputs=y.shape[1], 
                hidden=np.repeat(128, width), 
                learning_rate=0.001, 
                l1_penalty=0,
            )
            self.nnet.fit(X, y, epochs=500, batch_size=16, workers=-1, verbose=0)

    def predict(self, X):
        print("Model Prediction:")
        start = time.time()

        # transform and predict new data
        print("> Transforming The New Data")
        X = self.names1.transform(X)
        X = self.datetime.transform(X)
        X = self.categorical.transform(X)
        X = self.names2.transform(X)
        X = self.impute.transform(X)
        X = self.constant1.transform(X)
        X = self.selection1.transform(X)
        X = self.scaler1.transform(X)
        numbers = self.numbers.transform(X)
        X = self.bin.transform(X)
        X = self.reciprocals.transform(X)
        X = self.interactions.transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.transform(X)
        X = self.selection2.transform(X)
        X = self.scaler2.transform(X)
        print("> Getting Predictions")
        y = self.nnet.predict(X, verbose=0).ravel()

        end = time.time()
        self.run_time(start, end)

        print("Model Monitoring:")
        start = time.time()

        print("> Computing Feature Drift")
        self.monitor(X, y)

        end = time.time()
        self.run_time(start, end)

        return y
    
    def refit(self, X, y):
        # add the new data to the model data
        self.X = pd.concat([self.X, X], axis="index").reset_index(drop=True)
        self.y = pd.concat([self.y, y], axis="index").reset_index(drop=True)

        print("Model Retraining:")
        start = time.time()

        # transform the new data
        print("> Transforming The Updated Data")
        X = self.names1.fit_transform(self.X)
        X = self.datetime.fit_transform(X)
        X = self.categorical.fit_transform(X)
        X = self.names2.fit_transform(X)
        X = self.impute.fit_transform(X)
        X = self.constant1.fit_transform(X)
        X = self.selection1.fit_transform(X, self.y)
        X = self.scaler1.fit_transform(X)
        numbers = self.numbers.fit_transform(X)
        X = self.bin.fit_transform(X)
        X = self.reciprocals.fit_transform(X)
        X = self.interactions.fit_transform(X)
        X = pd.concat([X, numbers], axis="columns")
        X = self.constant2.fit_transform(X)
        X = self.selection2.fit_transform(X, self.y)
        X = self.scaler2.fit_transform(X)

        print("> Training Neural Network")
        if self.deep:
            width = 10
        else:
            width = 2
        self.nnet = self.build(
            inputs=X.shape[1], 
            outputs=self.y.shape[1], 
            hidden=np.repeat(128, width), 
            learning_rate=0.001, 
            l1_penalty=0,
        )
        self.nnet.fit(X, self.y, epochs=500, batch_size=16, workers=-1, verbose=0)

        end = time.time()
        self.run_time(start, end)

        print("Model Indicators:")
        start = time.time()
        
        print("> Perturbing Features")
        self.importance(X, self.y)

        end = time.time()
        self.run_time(start, end)

    def performance(self, X, y):
        # compute RMSE and R2
        predictions = self.nnet.predict(X, verbose=0).ravel()
        y = y.iloc[:,0].to_numpy()
        self.bootstrap(y, predictions)
        df = pd.DataFrame({
            "RMSE": self.rmse,
            "R2": self.r2,
        })
        self.rmse = np.mean(self.rmse)
        self.r2 = np.mean(self.r2)

        # plot RMSE and R2
        if self.plots:
            self.histogram(
                df,
                x="RMSE",
                bins=20,
                title="Histogram For RMSE",
                font_size=16,
            )
            self.histogram(
                df,
                x="R2",
                bins=20,
                title="Histogram For R2",
                font_size=16,
            )

        # compute control limits for residuals
        error = y - predictions
        df = self.imr(error)

        # plot the control limits for residuals
        in_control = df.loc[(df["Individual"] >= df["Individual LCL"]) & (df["Individual"] <= df["Individual UCL"])].shape[0]
        in_control /= df.shape[0]
        in_control *= 100
        in_control = f"{round(in_control, 2)}%"
        if self.plots:
            self.histogram(
                df,
                x="Individual",
                vlines=[df["Individual LCL"][0], df["Individual UCL"][0]],
                bins=20,
                title=f"Histogram For Residuals, {in_control} In Control",
                font_size=16,
            )
        self.in_control = in_control

        # plot the control limits for the moving range of residuals
        in_control = df.loc[(df["Moving Range"] >= df["Moving Range LCL"]) & (df["Moving Range"] <= df["Moving Range UCL"])].shape[0]
        in_control /= df.shape[0]
        in_control *= 100
        in_control = f"{round(in_control, 2)}%"
        if self.plots:
            self.histogram(
                df,
                x="Moving Range",
                vlines=[df["Moving Range LCL"][0], df["Moving Range UCL"][0]],
                bins=20,
                title=f"Histogram For The Moving Range Of Residuals, {in_control} In Control",
                font_size=16,
            )

        # plot the predictions
        df = pd.DataFrame({
            "Prediction": predictions,
            "Actual": y,
        })
        if self.plots:
            self.parity(
                df,
                predict="Prediction",
                actual="Actual",
                title="Parity Plot",
                font_size=16,
            )

    def importance(self, X, y):
        # get the feature importance to determine indicators of the target
        # set up random numbers for perturbing features
        iterations = 1  # how many times to perturb each feature
        np.random.seed(0)
        perturb1 = np.random.uniform(
            low=0.9, 
            high=1.1, 
            size=X.shape[0] * iterations,
        ).reshape(X.shape[0], iterations)
        perturb2 = np.random.uniform(
            low=-0.1, 
            high=0.1, 
            size=X.shape[0] * iterations,
        ).reshape(X.shape[0], iterations)
        rmse = list()
        indicators = pd.DataFrame()
        X_perturb = X.copy()
        
        # get the baseline prediction error
        baseline = mean_squared_error(
            y_true=y.iloc[:,0].to_numpy(),
            y_pred=self.nnet.predict(X, verbose=0).ravel(),
            squared=False,
        )

        # perturb each feature to evaluate how much the prediction error changes
        for col in X.columns:
            mean = X[col].mean()
            for i in range(iterations):
                X_perturb[col] = X[col] * perturb1[:,i] + perturb2[:,i]  # perturb
                rmse.append(mean_squared_error(  # prediction error
                    y_true=y.iloc[:,0].to_numpy(),
                    y_pred=self.nnet.predict(X_perturb, verbose=0).ravel(),
                    squared=False,
                ))
            
            # feature importance
            rmse = np.mean(rmse)
            change = rmse / baseline - 1  # if the rmse changes a lot then it must be an important feature
            indicators = pd.concat([
                indicators, 
                pd.DataFrame({
                    "Indicator": [col], 
                    "Importance": [change],
                }),
            ], axis="index").reset_index(drop=True)
            rmse = list()

        indicators = indicators.sort_values(
            by="Importance", 
            ascending=False,
        ).reset_index(drop=True)
        indicators = indicators.loc[indicators["Importance"] > 0]

        # plot the feature importance
        if self.plots:
            self.bar_plot(
                indicators,
                x="Indicator",
                y="Importance",
                title="Feature Importance",
                font_size=16,
            )
        self.indicators = indicators
        
    def monitor(self, X, y):
        y_name = self.y.columns[0]
        X[y_name] = y  # new data

        # transform the raw data
        modelX = self.names1.transform(self.X)
        modelX = self.datetime.transform(modelX)
        modelX = self.categorical.transform(modelX)
        modelX = self.names2.transform(modelX)
        modelX = self.impute.transform(modelX)
        modelX = self.constant1.transform(modelX)
        modelX = self.selection1.transform(modelX)
        modelX = self.scaler1.transform(modelX)
        numbers = self.numbers.transform(modelX)
        modelX = self.bin.transform(modelX)
        modelX = self.reciprocals.transform(modelX)
        modelX = self.interactions.transform(modelX)
        modelX = pd.concat([modelX, numbers], axis="columns")
        modelX = self.constant2.transform(modelX)
        modelX = self.selection2.transform(modelX)
        modelX = self.scaler2.transform(modelX)
        df = pd.concat([modelX, self.y], axis="columns")  # data we trained on

        # see if the distribtuion of the new data is the same as the data we trained on
        pvalues = list()
        for column in df.columns:
            pvalues.append(kstest(
                df[column].tolist(),
                X[column].tolist(),
            ).pvalue)
        pvalues = pd.DataFrame({
            "Feature": df.columns,
            "pvalue": pvalues,
        })
        pvalues = pvalues.sort_values(
            by="pvalue", 
            ascending=False,
        ).reset_index(drop=True)

        # plot the pvalues
        if self.plots:
            self.bar_plot(
                pvalues,
                x="Feature",
                y="pvalue",
                title="Feature Drift, Drift Detected If pvalue < 0.05",
                font_size=16,
            )
        self.drift = pvalues

    def bootstrap(self, y_true, y_pred):
        df = pd.DataFrame({
            "Actual": y_true,
            "Predict": y_pred,
        })

        self.rmse = list()
        self.r2 = list()
        np.random.seed(0)
        seeds = np.random.random_integers(low=0, high=1e6, size=1000)

        # randomly sample RMSE and R2 scores
        for i in range(1000):
            sample = df.sample(frac=0.5, replace=True, random_state=seeds[i])
            self.rmse.append(mean_squared_error(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
                squared=False,
            ))
            self.r2.append(r2_score(
                y_true=sample["Actual"].tolist(),
                y_pred=sample["Predict"].tolist(),
            ))

    def imr(self, x: list):
        # control chart constants
        d2 = 1.128
        D4 = 3.267
        D3 = 0

        data = pd.DataFrame({
            "Observation": np.arange(len(x)) + 1,
            "Individual": x,
        })
        data["Moving Range"] = data["Individual"].diff().abs()
        data = data.dropna().reset_index(drop=True)

        # center lines
        Xbar = data["Individual"].mean()
        MRbar = data["Moving Range"].mean()

        # control limits
        I_UCL = Xbar + 3*MRbar / d2
        I_LCL = Xbar - 3*MRbar / d2
        I_CL = Xbar

        MR_UCL = MRbar*D4
        MR_LCL = MRbar*D3
        MR_CL = MRbar

        # results
        df = data[["Observation"]].copy()
        df["Individual"] = data["Individual"]
        df["Individual UCL"] = I_UCL
        df["Individual LCL"] = I_LCL
        df["Individual CL"] = I_CL
        df["Moving Range"] = data["Moving Range"]
        df["Moving Range UCL"] = MR_UCL
        df["Moving Range LCL"] = MR_LCL
        df["Moving Range CL"] = MR_CL

        return df
    
    def build(self, inputs, outputs, hidden, learning_rate=0.001, l1_penalty=0):
        activation = "linear"
        loss = "mean_squared_error"
        
        # build the network
        inputs = keras.Input(shape=(inputs,))
        layer = layers.Dense(
            units=hidden[0], 
            activation="relu",
            kernel_regularizer=regularizers.l1(l1_penalty),
        )(inputs)
        for i in range(1, len(hidden)):
            dense = layers.Dense(
                units=hidden[i], 
                activation="relu",
                kernel_regularizer=regularizers.l1(l1_penalty),
            )
            layer = dense(layer)
        outputs = layers.Dense(units=outputs, activation=activation)(layer)

        # compile the model
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=[loss for _ in hidden],
            optimizer=optimizers.Adam(lr=learning_rate),
        )
        return model

    def parity(self, df, predict, actual, color=None, title="Parity Plot", font_size=None):
        fig = px.scatter(df, x=actual, y=predict, color=color, title=title)
        fig.add_trace(go.Scatter(x=df[actual], y=df[actual], mode="lines", showlegend=False, name="Actual"))
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}plots{path_sep}{title}.html")
    
    def histogram(self, df, x, bins=20, vlines=None, title="Histogram", font_size=None):
        bin_size = (df[x].max() - df[x].min()) / bins
        fig = px.histogram(df, x=x, title=title)
        if vlines is not None:
            for line in vlines:
                fig.add_vline(x=line)
        fig.update_traces(xbins=dict( # bins used for histogram
                size=bin_size,
            ))
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}plots{path_sep}{title}.html")

    def bar_plot(self, df, x, y, color=None, title="Bar Plot", font_size=None):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}plots{path_sep}{title}.html")

    def run_time(self, start, end):
        duration = end - start
        if duration < 60:
            duration = f"{round(duration, 2)} Seconds"
        elif duration < 3600:
            duration = f"{round(duration / 60, 2)} Minutes"
        else:
            duration = f"{round(duration / 3600, 2)} Hours"
        print(duration)

    def folder(self, name):
        if not os.path.isdir(name):
            os.mkdir(name)

    def dump(self):
        # save the machine learning pipeline and data
        # fit() or validate() has to be called for the pipeline and indicators to exist
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names1", "wb") as f:
            pickle.dump(self.names1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}datetime", "wb") as f:
            pickle.dump(self.datetime, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}categorical", "wb") as f:
            pickle.dump(self.categorical, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names2", "wb") as f:
            pickle.dump(self.names2, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}impute", "wb") as f:
            pickle.dump(self.impute, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant1", "wb") as f:
            pickle.dump(self.constant1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection1", "wb") as f:
            pickle.dump(self.selection1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler1", "wb") as f:
            pickle.dump(self.scaler1, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}numbers", "wb") as f:
            pickle.dump(self.numbers, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}bin", "wb") as f:
            pickle.dump(self.bin, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}reciprocals", "wb") as f:
            pickle.dump(self.reciprocals, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}interactions", "wb") as f:
            pickle.dump(self.interactions, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant2", "wb") as f:
            pickle.dump(self.constant2, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection2", "wb") as f:
            pickle.dump(self.selection2, f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler2", "wb") as f:
            pickle.dump(self.scaler2, f)
        self.nnet.save(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}nnet.keras")
        self.X.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}X.csv", index=False)
        self.y.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}y.csv", index=False)
        self.indicators.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}indicators.csv", index=False)
        try:  # predict() has to be called for drift to exist
            self.drift.to_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}drift.csv", index=False)
        except:
            pass
        try:  # validate() has to be called for rmse, r2, and in_control to exist
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}rmse", "wb") as f:
                pickle.dump(self.rmse, f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}r2", "wb") as f:
                pickle.dump(self.r2, f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}in_control", "wb") as f:
                pickle.dump(self.in_control, f)
        except:
            pass

    def load(self):
        # load the machine learning pipeline and data
        # fit() or validate() had to have been called for the pipeline and indicators to exist
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names1", "rb") as f:
            self.names1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}datetime", "rb") as f:
            self.datetime = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}categorical", "rb") as f:
            self.categorical = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}names2", "rb") as f:
            self.names2 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}impute", "rb") as f:
            self.impute = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant1", "rb") as f:
            self.constant1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection1", "rb") as f:
            self.selection1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler1", "rb") as f:
            self.scaler1 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}numbers", "rb") as f:
            self.numbers = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}bin", "rb") as f:
            self.bin = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}reciprocals", "rb") as f:
            self.reciprocals = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}interactions", "rb") as f:
            self.interactions = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}constant2", "rb") as f:
            self.constant2 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}selection2", "rb") as f:
            self.selection2 = pickle.load(f)
        with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}scaler2", "rb") as f:
            self.scaler2 = pickle.load(f)
        self.nnet = keras.models.load_model(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}nnet.keras")
        self.X = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}X.csv")
        self.y = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}y.csv")
        self.indicators = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}indicators.csv")
        try:  # predict() had to have been called for drift to exist
            self.drift = pd.read_csv(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}drift.csv")
        except:
            pass
        try:  # validate() had to have been called for rmse, r2, and in_control to exist
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}rmse", "rb") as f:
                self.rmse = pickle.load(f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}r2", "rb") as f:
                self.r2 = pickle.load(f)
            with open(f"{self.path}{path_sep}{self.name}{path_sep}dump{path_sep}in_control", "rb") as f:
                self.in_control = pickle.load(f)
        except:
            pass


class FeatureNames:
    def __init__(self, rename=True, verbose=True):
        self.rename = rename
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.rename:
            return self
        if self.verbose:
            print("> Renaming Features")

        self.columns = [re.sub(" ", "_", col) for col in X.columns]
        return self

    def transform(self, X, y=None):
        if not self.rename:
            return X

        X = X.copy()
        X.columns = self.columns
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class TimeFeatures:
    def __init__(self, time=True, verbose=True):
        self.time = time
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.time:
            return self
        if self.verbose:
            print("> Extracting Time Features")

        # convert any timestamp columns to datetime data type
        df = X.apply(
            lambda col: pd.to_datetime(col, errors="ignore")
            if col.dtypes == object 
            else col, 
            axis=0,
        )

        # check if any columns are timestamps
        self.features = [col for col in df.columns if is_datetime(df[col])]
        return self

    def transform(self, X, y=None):
        if not self.time:
            return X

        if len(self.features) == 0:
            return X
        else:
            X = X.copy()

            # convert any timestamp columns to datetime data type
            df = X[self.features].apply(
                lambda col: pd.to_datetime(col, errors="ignore")
                if col.dtypes == object 
                else col, 
                axis=0,
            )

            # extract timestamp features
            dt = pd.DataFrame()
            for col in self.features:
                # timestamp components
                dt[f"{col}_year"] = df[col].dt.isocalendar().year
                dt[f"{col}_quarter"] = df[col].dt.quarter
                dt[f"{col}_month"] = df[col].dt.month
                dt[f"{col}_week"] = df[col].dt.isocalendar().week
                dt[f"{col}_day_of_month"] = df[col].dt.day
                dt[f"{col}_day_of_week"] = df[col].dt.day_name()
                dt[f"{col}_hour"] = df[col].dt.hour
                dt[f"{col}_minute"] = df[col].dt.minute
                
                # economic data
                dates = df[col].dt.date
                start = min(dates)
                end = max(dates)
                fred = pdr.DataReader([
                    "NASDAQCOM", 
                    "UNRATE", 
                    "CPALTT01USM657N", 
                    "PPIACO",
                    "GDP",
                    "GDI",
                    "FEDFUNDS",
                ], "fred", start, end).reset_index()
                seq = pd.DataFrame({"DATE": pd.date_range(start=start, end=end)})
                fred = seq.merge(right=fred, how="left", on="DATE")
                fred = fred.ffill().bfill()  # fill in missing values with the last known value
                dt_fred = pd.DataFrame({"DATE": pd.to_datetime(dates)})
                dt_fred = dt_fred.merge(right=fred, how="left", on="DATE")
                dt[f"{col}_nasdaq"] = dt_fred["NASDAQCOM"]
                dt[f"{col}_unemployment"] = dt_fred["UNRATE"]
                dt[f"{col}_cpi"] = dt_fred["CPALTT01USM657N"]
                dt[f"{col}_ppi"] = dt_fred["PPIACO"]
                dt[f"{col}_gdp"] = dt_fred["GDP"]
                dt[f"{col}_gdi"] = dt_fred["GDI"]
                dt[f"{col}_federal_funds_rate"] = dt_fred["FEDFUNDS"]

            dt = pd.concat([X.drop(columns=self.features), dt], axis="columns")
            return dt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class CategoricalFeatures:
    def __init__(self, binary=True, verbose=True):
        self.binary = binary
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.binary:
            return self
        if self.verbose:
            print("> Transforming Categorical Features")

        strings = X.select_dtypes(include="object").columns.tolist()
        df = X.copy().drop(columns=strings)
        numbers = [col for col in df.columns if len(df[col].unique()) <= 60 and (df[col] % 1 == 0).all()]
        self.categorical = strings + numbers
        if len(self.categorical) == 0:
            return self
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        return self.encoder.fit(X[self.categorical].astype(str))

    def transform(self, X, y=None):
        if not self.binary:
            return X

        if len(self.categorical) == 0:
            return X
        continuous = X.copy().drop(columns=self.categorical)
        binary = self.encoder.transform(X[self.categorical].astype(str))
        binary = pd.DataFrame(binary, columns=self.encoder.get_feature_names_out())
        df = pd.concat([continuous, binary], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ImputeFeatures:
    def __init__(self, imputation=True, verbose=True):
        self.imputation = imputation
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.imputation:
            return self
        if self.verbose:
            print("> Filling In Missing Values")

        self.columns = X.columns
        self.imputer = KNNImputer()
        return self.imputer.fit(X)

    def transform(self, X, y=None):
        if not self.imputation:
            return X

        df = self.imputer.transform(X)
        df = pd.DataFrame(df, columns=self.columns)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ConstantFeatures:
    def __init__(self, variance=True, verbose=True):
        self.variance = variance
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.variance:
            return self
        if self.verbose:
            print("> Removing Constant Features")

        self.selector = VarianceThreshold()
        return self.selector.fit(X)

    def transform(self, X, y=None):
        if not self.variance:
            return X

        df = self.selector.transform(X)
        df = pd.DataFrame(df, columns=self.selector.get_feature_names_out())
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class ScaleFeatures:
    def __init__(self, scale=True, bounds=(0, 1), verbose=True):
        self.scale = scale
        self.bounds = bounds
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.scale:
            return self
        if self.verbose:
            print("> Scaling Features")
        
        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        if len(self.columns) == 0:
            return self
        self.scaler = MinMaxScaler(feature_range=self.bounds)
        return self.scaler.fit(X[self.columns])

    def transform(self, X, y=None):
        if not self.scale:
            return X

        if len(self.columns) == 0:
            return X
        df = self.scaler.transform(X[self.columns])
        df = pd.DataFrame(df, columns=self.columns)
        df = pd.concat([X.drop(columns=self.columns), df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    
class AtwoodNumbers:
    def __init__(self, atwood=True, verbose=True):
        self.atwood = atwood
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.atwood:
            return self
        if self.verbose:
            print("> Computing Atwoood Numbers")

        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        return self

    def transform(self, X, y=None):
        if not self.atwood:
            return pd.DataFrame()

        if len(self.columns) < 2:
            return pd.DataFrame()
        numbers = list()
        pairs = list(combinations(self.columns, 2))
        for pair in pairs:
            numbers.append(pd.DataFrame({
                f"({pair[0]}-{pair[1]})/({pair[0]}+{pair[1]})": (X[pair[0]] - X[pair[1]]) / (X[pair[0]] + X[pair[1]]),
            }))
        df = pd.concat(numbers, axis="columns")
        df = df.fillna(0)
        df.replace(np.inf, 1e6, inplace=True)
        df.replace(-np.inf, -1e6, inplace=True)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    

class BinFeatures:
    def __init__(self, binning=True, verbose=True):
        self.binning = binning
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.binning:
            return self
        if self.verbose:
            print("> Binning Features")
        
        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        if len(self.columns) == 0:
            return self
        self.binner = KBinsDiscretizer(n_bins=3, encode="onehot", strategy="uniform", subsample=None)
        return self.binner.fit(X[self.columns])

    def transform(self, X, y=None):
        if not self.binning:
            return X

        if len(self.columns) == 0:
            return X
        df = self.binner.transform(X[self.columns]).toarray()
        edges = self.binner.bin_edges_
        columns = list()
        for i, feature in enumerate(self.columns):
            bins = np.around(edges[i], 6)
            columns.append(f"{feature}({bins[0]}-{bins[1]})")
            columns.append(f"{feature}({bins[1]}-{bins[2]})")
            columns.append(f"{feature}({bins[2]}-{bins[3]})")
        df = pd.DataFrame(df, columns=columns)
        df = pd.concat([X, df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
    
class Reciprocals:
    def __init__(self, reciprocal=True, verbose=True):
        self.reciprocal = reciprocal
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.reciprocal:
            return self
        if self.verbose:
            print("> Computing Reciprocals")

        self.columns = [col for col in X.columns if len(X[col].unique()) > 2]
        return self

    def transform(self, X, y=None):
        if not self.reciprocal:
            return X

        if len(self.columns) == 0:
            return X
        df = 1 / X.copy()[self.columns]
        df.replace(np.inf, 1e6, inplace=True)
        df.replace(-np.inf, -1e6, inplace=True)
        df.columns = [f"1/{col}" for col in df.columns]
        df = pd.concat([X, df], axis="columns")
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class Interactions:
    def __init__(self, interaction=True, verbose=True):
        self.interaction = interaction
        self.verbose = verbose

    def fit(self, X, y=None):
        if not self.interaction:
            return self
        if self.verbose:
            print("> Computing Interactions")

        self.interactions = PolynomialFeatures(
            degree=2, 
            interaction_only=True, 
            include_bias=False,
        )
        return self.interactions.fit(X)

    def transform(self, X, y=None):
        if not self.interaction:
            return X

        df = self.interactions.transform(X)
        columns = self.interactions.get_feature_names_out()
        columns = [re.sub(" ", "*", col) for col in columns]
        df = pd.DataFrame(df, columns=columns)
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class FeatureSelector:
    def __init__(self, selection=True, verbose=True):
        self.selection = selection
        self.verbose = verbose

    def fit(self, X, y):
        if not self.selection:
            return self
        if self.verbose:
            print("> Selecting Features")

        tree = XGBRegressor(
            booster="gbtree",
            n_estimators=25, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=0,
            n_jobs=-1,
        )
        tree.fit(X, y)

        # get the feature importance to determine indicators of the target
        importance = tree.feature_importances_
        indicators = pd.DataFrame({
            "Indicator": X.columns,
            "Importance": importance,
        })
        indicators = indicators.sort_values(
            by="Importance", 
            ascending=False,
        ).reset_index(drop=True)
        indicators = indicators.loc[indicators["Importance"] > 0]
        self.columns = indicators["Indicator"].tolist()

        return self

    def transform(self, X, y=None):
        if not self.selection:
            return X

        return X[self.columns]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
