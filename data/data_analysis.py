import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
Dataset: Bleaching and environmental data for global coral reef sites from 1980-2020
BCO-DMO
https://www.bco-dmo.org/dataset/773466

Variables:

Date - Year-Moth-Day
Temperature_Mean - Mean temperature in Kelvin
Temperature_Maximum - Max temperature in Kelvin
SSTA - Surface Sea Temperature Anomaly
SSTA_DHW - Sea Surface Temperature Degree Heating Weeks (duration and intesity)
Percent_Bleaching - Average of four transect segments (Reef Check) or average of a bleaching code

'''


df = pd.read_csv("global_bleaching_environmental.csv", low_memory=False)


columns = [
    "Date",
    "Temperature_Mean",
    "Temperature_Maximum",
    "Temperature_Minimum",
    "SSTA",
    "SSTA_DHW",
    "Percent_Bleaching"
]

df = df[columns]

print("\n------------------------------------")
print("DATA")
print("------------------------------------")

print(df.head())


#convert date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


#make values numeric or NaN
numeric_columns = [
    "Temperature_Mean",
    "Temperature_Maximum",
    "Temperature_Minimum",
    "SSTA",
    "SSTA_DHW",
    "Percent_Bleaching"
] 

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


#remove missing values
df = df.dropna()
print("\nData shape after removing all missing values: ", df.shape)


#start of bleaching
print("\n------------------------------------")
print("MEANS AT START OF BLEACHING")
print("------------------------------------")
bleaching_started = df[df["Percent_Bleaching"] > 0]
avg_temp_bleaching_start = round(bleaching_started["Temperature_Mean"].mean(), 2)
avg_max_temp_bleaching_start = round(bleaching_started["Temperature_Maximum"].mean(), 2)
avg_min_temp_bleaching_start = round(bleaching_started["Temperature_Minimum"].mean(), 2)
avg_SSTA_bleaching_start = round(bleaching_started["SSTA"].mean(), 2)
avg_SSTA_DHW_bleaching_start = round(bleaching_started["SSTA_DHW"].mean(), 2)


print("\nAverage temperature (Kelvin) when bleaching starts: ", avg_temp_bleaching_start)
print("\nAverage maximum temperature (Kelvin) when bleaching starts: ", avg_max_temp_bleaching_start)
print("\nAverage minimum temperature (Kelvin) when bleaching starts: ", avg_min_temp_bleaching_start)
print("\nAverage SSTA when bleaching starts: ", avg_SSTA_bleaching_start)
print("\nAverage SSTA_DHW when bleaching starts: ", avg_SSTA_DHW_bleaching_start)


print("\n------------------------------------")
print("PLOTS")
print("------------------------------------")

#plot1: mean temperature vs. percent bleaching
plt.figure()
plt.scatter(df["Temperature_Mean"], df["Percent_Bleaching"], alpha=0.5)
plt.xlabel("Mean Water Temperature (Kelvin)")
plt.ylabel("Percent Bleaching")
plt.title("Mean Temperature vs Bleaching Percentage")
plt.show()


#plot2: DHW vs. bleaching percent
plt.figure()
plt.scatter(df["SSTA_DHW"], df["Percent_Bleaching"], alpha=0.5)
plt.xlabel("Degree Heating Weeks (DHW)")
plt.ylabel("Percent Bleaching")
plt.title("Heat Stress Duration vs Bleaching Percentage")
plt.show()


#plot3: grouped DHW
bins = np.arange(0, df["SSTA_DHW"].max() + 1, 1)
df["DHW_bin"] = pd.cut(df["SSTA_DHW"], bins=bins)
grouped = df.groupby("DHW_bin")["Percent_Bleaching"].mean()

plt.figure()
grouped.plot(marker="o")
plt.xlabel("Degree Heating Weeks")
plt.ylabel("Mean Percent Bleaching")
plt.title("Heat Stress Duration vs Mean Bleaching Percentage")
plt.show()


