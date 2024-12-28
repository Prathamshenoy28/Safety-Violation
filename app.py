import streamlit as st
import pandas as pd
import json

st.title("Safety Violation Dashboard")
st.subheader("Detected Violations in the Processed Video")


try:
    with open('violations.json', 'r') as f:
        violation_events = json.load(f)
except FileNotFoundError:
    st.warning("No violations data found. Process the video first.")
    violation_events = []


if violation_events:
    df = pd.DataFrame(violation_events)
    st.table(df)

    
    st.subheader("Filter Violations")
    filter_violation = st.selectbox("Select Violation Type:", options=["All"] + list(df["violation_type"].unique()))
    
    if filter_violation != "All":
        filtered_df = df[df["violation_type"] == filter_violation]
        st.write(f"Filtered by: {filter_violation}")
        st.table(filtered_df)
    else:
        st.write("Showing all violations")

else:
    st.info("No violations detected.")


import streamlit as st
import os


video_path = 'C:/Users/HP/Downloads/your_output_directory2/output_video_reencoded.mp4'


if os.path.exists(video_path):
    st.video(video_path)
else:
    st.error(f"Video file not found at {video_path}")

