import streamlit as st
import pandas as pd
import os
from PIL import Image
import time

LOG_CSV = os.path.join('logs', 'violations.csv')
LATEST_IMG = os.path.join('logs', 'latest.jpg')

st.title('AI-Based MMS Safety Violation Dashboard')

# Live video snapshot (auto-refresh every second)
st.subheader('Live Video Feed (Snapshot)')
if os.path.exists(LATEST_IMG):
    st.image(LATEST_IMG, caption='Live Video Snapshot', use_column_width=True)
    st.caption('This image auto-refreshes every second.')
    st_autorefresh = st.empty()
    time.sleep(1)
else:
    st.write('No live video available.')

# Load violation logs
def load_logs():
    if os.path.exists(LOG_CSV):
        return pd.read_csv(LOG_CSV)
    else:
        return pd.DataFrame(columns=['timestamp', 'violation_type', 'image_path'])

st.write("To refresh, click the 'Rerun' button in the Streamlit menu (top right).")

logs = load_logs()

# Filtering
violation_types = ['All'] + sorted(logs['violation_type'].unique()) if not logs.empty else ['All']
selected_type = st.selectbox('Filter by Violation Type', violation_types)
if selected_type != 'All':
    logs = logs[logs['violation_type'] == selected_type]

st.subheader('Violation Log')
st.dataframe(logs)

# Show latest violation image
if not logs.empty:
    latest = logs.iloc[-1]
    st.subheader(f"Latest Violation: {latest['violation_type']} at {latest['timestamp']}")
    if os.path.exists(latest['image_path']):
        img = Image.open(latest['image_path'])
        st.image(img, caption='Latest Violation Snapshot', use_column_width=True)
    else:
        st.write('Image not found.')
else:
    st.write('No violations logged yet.') 