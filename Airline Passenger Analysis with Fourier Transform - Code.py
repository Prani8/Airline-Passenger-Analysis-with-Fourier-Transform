#!/usr/bin/env python
# coding: utf-8

# # Small project - 23095964

# ### Import necessary libraries

# In[4]:


import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt


# ### Load the dataset

# In[7]:


file_path = 'airline6.csv'
airline_data = pd.read_csv(file_path)


# ### Convert 'Date' to datetime and extract

# In[10]:


airline_data['Date'] = pd.to_datetime(airline_data['Date'])
airline_data['Month'] = airline_data['Date'].dt.month
airline_data['Year'] = airline_data['Date'].dt.year


# ### Fourier Transform on daily passenger numbers

# In[33]:


passenger_numbers = airline_data['Number'].values
fourier_transform = fft(passenger_numbers)
frequencies = np.fft.fftfreq(len(passenger_numbers), d=1)


# ### Calculate average daily passengers for each month

# In[22]:


monthly_avg_passengers = airline_data.groupby('Month')['Number'].mean()


# ### Approximate monthly passenger numbers using the first 8 Fourier termsd

# In[52]:


n_terms = 8
fourier_approx = np.zeros(len(passenger_numbers), dtype=complex)

for k in range(n_terms):
    fourier_approx += fourier_transform[k] * np.exp(2j * np.pi * frequencies[k] * np.arange(len(passenger_numbers)))

fourier_approx_real = np.real(fourier_approx)

# Fourier series to monthly average levels
monthly_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
cumulative_days = np.cumsum([0] + monthly_days)

# Monthly averages from Fourier approximation##
fourier_monthly_avg = [
    np.mean(fourier_approx_real[cumulative_days[i]:cumulative_days[i+1]]) for i in range(len(monthly_days))
]

scaling_factor = np.mean(monthly_avg_passengers) / np.mean(fourier_monthly_avg)
fourier_monthly_avg_scaled = np.array(fourier_monthly_avg) * scaling_factor


# ### Plot the monthly average and Fourier approximation

# In[50]:


plt.figure(figsize=(12, 6))
plt.bar(monthly_avg_passengers.index, monthly_avg_passengers.values, color='skyblue', label='Monthly Avg Passengers')
plt.plot(np.arange(1, 13), fourier_monthly_avg_scaled, color='orange', label='Fourier Approximation', linewidth=2, marker='o')

plt.title('Average Daily Number of Passengers Flown (2021-2022)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Number of Passengers', fontsize=12)
plt.xticks(ticks=np.arange(1, 13), labels=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

# Student ID label
plt.text(2.7, max(monthly_avg_passengers.values) * 0.89, "Student ID: 23095964", fontsize=12, color='red', ha='right')
plt.show()


# ### Calculate the Power Spectrum from Fourier Transform

# In[48]:


power_spectrum = np.abs(fourier_transform) ** 2
periods = 1 / frequencies[1:len(frequencies)//2]

# Filter relevant periods between 7 and 365 days
valid_indices = np.where((periods >= 7) & (periods <= 365))
filtered_periods = periods[valid_indices]
filtered_power = power_spectrum[1:len(frequencies)//2][valid_indices]

# Average ticket prices for each year
avg_price_2021 = airline_data[airline_data['Year'] == 2021]['Revenue'].mean()
avg_price_2022 = airline_data[airline_data['Year'] == 2022]['Revenue'].mean()


# ### Plot the Power Spectrum

# In[46]:


plt.figure(figsize=(12, 6))
plt.plot(filtered_periods, filtered_power, color='blue', label='Power Spectrum', linewidth=2)

plt.title('Power Spectrum of Daily Passenger Number Variation', fontsize=14)
plt.xlabel('Period (Days)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# Student ID and ticket prices
plt.text(max(filtered_periods) * 0.2, max(filtered_power) * 0.7, f"2021 Avg Price (X): £{avg_price_2021:.2f}", fontsize=12, color='green', ha='center')
plt.text(max(filtered_periods) * 0.2, max(filtered_power) * 0.6, f"2022 Avg Price (Y): £{avg_price_2022:.2f}", fontsize=12, color='green', ha='center')
plt.text(max(filtered_periods) * 0.5, max(filtered_power) * 0.9, "Student ID: 23095964", fontsize=12, color='red', ha='left')

plt.show()


# # Insights from the data:

# #### 1. Data Preparation: After loading the dataset, crucial time information such as months and years is taken out to facilitate analysis.
# #### 2. Seasonal Pattern Detection: To find recurring patterns, such as monthly and annual passenger trends, a Fourier transform is utilised.
# #### 3. Monthly Passenger Trends: To identify seasonal highs and lows, monthly averages are computed and compared using a mathematical model.
# #### 4. Insights into Travel Cycles: The power spectrum displays prominent travel cycles, including hectic holiday seasons or consistent monthly variations.
# #### 5. Comparison of Ticket Prices: Comparing the average ticket prices for 2021 and 2022 provides information on changes in revenue.
# #### 6. Clear Visuals: Passenger trends and seasonal cycles are visually represented by simple bar and line charts.
# #### 7. Beneficial Annotations: For improved display and context, charts are labelled with the student ID, ticket costs, and legends.
# #### 8. Business Insights: The research supports data-driven decision-making by highlighting travel trends, peak seasons, and potential causes of income swings.

# In[ ]:




