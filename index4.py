import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import os

# Get current timestamp and format date and time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

# Automatically refresh the Streamlit app every 2 seconds
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

# Display FizzBuzz based on count
if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Prepare attendance file path
attendance_file_path = f"Attendance/Attendance_{date}.csv"

# Check if the attendance file exists
if os.path.exists(attendance_file_path):
    # Read the attendance file and display it
    df = pd.read_csv(attendance_file_path)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    # Create an empty DataFrame and save it as the attendance file
    empty_df = pd.DataFrame(columns=["Name", "Status"])  # Adjust columns as needed
    empty_df.to_csv(attendance_file_path, index=False)
    st.write(f"Attendance file created for date: {date}. Please ensure attendance has been taken.")
