#################################### MusiScaleApp ###########################
import streamlit as st


# # Define the pages
main_page = st.Page("pages/Welcome.py",title="Welcome", default=True)
page_1 = st.Page("pages/FormalDiagramApp.py", title="Formal Diagram")
page_2 = st.Page("pages/TimeDurationDiagram.py", title="Time-Duration Diagram")
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Set up navigation
pg = st.navigation([main_page, page_1, page_2])

# Run the selected page
pg.run()
