def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import time
import numpy as np
import pandas as pd
import pandas_datareader as pdr


def prepare(df):
    data = Prepare(df)
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

    return data.df
    
    
class Prepare:
    def __init__(self, df):
        self.df = df  # dataset

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

    def run_time(self, start, end):
        duration = end - start
        if duration < 60:
            duration = f"{round(duration, 2)} Seconds"
        elif duration < 3600:
            duration = f"{round(duration / 60, 2)} Minutes"
        else:
            duration = f"{round(duration / 3600, 2)} Hours"
        print(duration)
