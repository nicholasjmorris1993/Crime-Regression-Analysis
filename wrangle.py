def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
import numpy as np
import pandas as pd
from itertools import combinations
import pandas_datareader as pdr
import scipy.cluster.hierarchy as sch
import plotly.express as px
from plotly.offline import plot

if os.name == "nt":
    path_sep = "\\"
else:
    path_sep = "/"


def prepare(df, name="Data Preparation", path=None, plots=True):
    data = Prepare(df, name, path, plots)
    print("Data Wrangling:")
    start = time.time()
    data.datetime()
    data.crime_rate()
    data.no_crime()
    data.time_features()
    data.economic_data()
    data.weekly_crime_rate()
    data.crime_rate_lags()
    end = time.time()
    data.run_time(start, end)
    
    if plots:
        print("Plotting:")
        start = time.time()
        data.separate()
        data.correlations()
        data.scatter_plots()
        data.bar_plots()
        data.pairwise_bar_plots()
        data.boxplots()
        end = time.time()
        data.run_time(start, end)

    return data.df
    
    
class Prepare:
    def __init__(
        self, 
        df,
        name="Data Preparation", 
        path=None,
        plots=True,
    ):
        self.df = df  # dataset
        self.name = name  # name of the analysis
        self.path = path  # the path where results will be exported
        self.plots = plots  # should we plot the analysis?
        
        if self.path is None:
            self.path = os.getcwd()

        # create folders for output files
        if self.plots:
            self.folder(f"{self.path}{path_sep}{self.name}")
    
    def datetime(self):
        print("> Converting Timestamps")
        # convert timestamps to datetime
        self.df["Dispatch_Date_Time"] = pd.to_datetime(self.df["Dispatch_Date_Time"])
        self.df["Dispatch_Date"] = pd.to_datetime(self.df["Dispatch_Date"])

        # sort the data by time and district
        self.df = self.df.sort_values(by=["Dispatch_Date_Time", "Dc_Dist"], ascending=True).reset_index(drop=True)

    def crime_rate(self):
        print("> Computing Crime Rate")
        # count how many crimes happened by day and district
        self.df = self.df.groupby(["Dispatch_Date", "Dc_Dist"]).agg({"Dispatch_Date": "count"})
        self.df.columns = ["Crimes"]
        self.df = self.df.reset_index()

    def no_crime(self):
        print("> Finding Days With No Crime")
        # get the days and places when there was no crime
        # create a grid of all day and district combinations
        days = pd.unique(self.df["Dispatch_Date"]).astype(str)
        districts = pd.unique(self.df["Dc_Dist"]).astype(str)
        grid = np.array(np.meshgrid(days, districts)).reshape(2, len(days) * len(districts)).T
        grid = pd.DataFrame(grid, columns=["Dispatch_Date", "Dc_Dist"])
        grid["Dispatch_Date"] = pd.to_datetime(grid["Dispatch_Date"])
        grid["Dc_Dist"] = grid["Dc_Dist"].astype(int)

        # join the crime rate onto the grid
        self.df = grid.merge(right=self.df, how="left", on=["Dispatch_Date", "Dc_Dist"])

        # replace missing values with 0
        self.df = self.df.fillna(0)

    def time_features(self):
        print("> Extracting Time Features")
        self.df["Year"] = self.df["Dispatch_Date"].dt.isocalendar().year
        self.df["Quarter"] = self.df["Dispatch_Date"].dt.quarter
        self.df["Month"] = self.df["Dispatch_Date"].dt.month
        self.df["Week"] = self.df["Dispatch_Date"].dt.isocalendar().week
        self.df["Year_Week"] = self.df["Year"].astype(str) + "_" + self.df["Week"].astype(str)

    def economic_data(self):
        print("> Getting Economic Data")
        dates = self.df["Dispatch_Date"].dt.date
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
        self.df["NASDAQ"] = dt_fred["NASDAQCOM"]
        self.df["Unemployment"] = dt_fred["UNRATE"]
        self.df["CPI"] = dt_fred["CPALTT01USM657N"]
        self.df["PPI"] = dt_fred["PPIACO"]
        self.df["GDP"] = dt_fred["GDP"]
        self.df["GDI"] = dt_fred["GDI"]
        self.df["Federal_Funds_Rate"] = dt_fred["FEDFUNDS"]
        
    def weekly_crime_rate(self):
        print("> Computing Weekly Crime Rate")
        self.df = self.df.groupby(["Year_Week", "Dc_Dist"]).agg({
            "Crimes": "sum", 
            "Year": "min", 
            "Quarter": "min", 
            "Month": "min", 
            "Week": "min", 
            "Dispatch_Date": "count",
            "NASDAQ": "mean",
            "Unemployment": "mean",
            "CPI": "mean",
            "PPI": "mean",
            "GDP": "mean",
            "GDI": "mean",
            "Federal_Funds_Rate": "mean",
        })
        self.df = self.df.rename(columns={"Dispatch_Date": "Days"})
        self.df = self.df.reset_index()

    def crime_rate_lags(self):
        print("> Computing Previous Weeks Of Crime")
        # sort by time and district
        self.df = self.df.sort_values(by=["Year", "Week", "Dc_Dist"], ascending=True).reset_index(drop=True)
        self.df = self.df.drop(columns="Year_Week")

        # for each district insert the previous month of crime
        crimes = pd.DataFrame()
        districts = pd.unique(self.df["Dc_Dist"])
        for dist in districts:
            data = self.df.loc[self.df["Dc_Dist"] == dist].reset_index(drop=True)
            for i in range(4):
                data[f"Crimes(t-{i+1})"] = data["Crimes"].shift(i+1)
            data = data.tail(data.shape[0] - 4).reset_index(drop=True)
            crimes = pd.concat([crimes, data], axis="index").reset_index(drop=True)
        self.df = crimes

        # sort by time and district
        self.df = self.df.sort_values(by=["Year", "Week", "Dc_Dist"], ascending=True).reset_index(drop=True)

    def separate(self):
        self.numbers = self.df.drop(columns=[
            "Dc_Dist", 
            "Year", 
            "Quarter", 
            "Month", 
            "Week", 
            "Days",
        ]).columns.tolist()
        self.strings = ["Dc_Dist", "Year", "Quarter", "Month", "Week", "Days"]

    def correlations(self):
        if self.plots:
            print("> Plotting Correlations")
            self.correlation_plot(
                df=self.df[self.numbers], 
                title="Correlation Heatmap",
                font_size=16,
            )

    def scatter_plots(self):
        if self.plots:
            pairs = list(combinations(self.numbers, 2))
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                self.scatter_plot(
                    df=self.df,
                    x=pair[0],
                    y=pair[1],
                    color=None,
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def histograms(self):
        if self.plots:
            for col in self.numbers:
                print(f"> Plotting {col}")
                self.histogram(
                    df=self.df,
                    x=col,
                    bins=20,
                    title=col,
                    font_size=16,
                )
                
    def bar_plots(self):
        if self.plots:
            for col in self.strings:
                print(f"> Plotting {col}")
                proportion = self.df[col].value_counts(normalize=True).reset_index()
                proportion.columns = ["Label", "Proportion"]
                proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)
                self.bar_plot(
                    df=proportion,
                    x="Proportion",
                    y="Label",
                    title=col,
                    font_size=16,
                )

    def pairwise_bar_plots(self):
        if self.plots:
            pairs = list(combinations(self.strings, 2))
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                data = pd.DataFrame()
                data[f"{pair[0]}, {pair[1]}"] = self.df[pair[0]].astype(str) + ", " + self.df[pair[1]].astype(str)
                proportion = data[f"{pair[0]}, {pair[1]}"].value_counts(normalize=True).reset_index()
                proportion.columns = [f"{pair[0]}, {pair[1]}", "Proportion"]
                proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)
                self.bar_plot(
                    df=proportion,
                    x="Proportion",
                    y=f"{pair[0]}, {pair[1]}",
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def boxplots(self):
        if self.plots:
            pairs = list()
            for number in self.numbers:
                for string in self.strings:
                    pairs.append((number, string))
            
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                # sort the data by the group average
                data = self.df.copy()
                df = data.groupby(pair[1]).agg({pair[0]: "mean"}).reset_index()
                df = df.sort_values(by=pair[0]).reset_index(drop=True).reset_index()
                df = df.drop(columns=pair[0])
                data = data.merge(right=df, how="left", on=pair[1])
                data = data.sort_values(by="index").reset_index(drop=True)
                self.box_plot(
                    df=data, 
                    x=pair[0], 
                    y=pair[1],
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def correlation_plot(self, df, title="Correlation Heatmap", font_size=None):
        df = df.copy()
        correlation = df.corr()

        # group columns together with hierarchical clustering
        X = correlation.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the correlation matrix for the received dataframe
        correlation = df.corr()

        # plot the correlation matrix
        fig = px.imshow(correlation, title=title, range_color=(-1, 1))
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def scatter_plot(self, df, x, y, color=None, title="Scatter Plot", font_size=None):
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def bar_plot(self, df, x, y, color=None, title="Bar Plot", font_size=None):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def box_plot(self, df, x, y, color=None, title="Box Plot", font_size=None):
        fig = px.box(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

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
