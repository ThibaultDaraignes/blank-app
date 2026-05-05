################# Welcome page (for MusiScale App) ##############

import streamlit as st

st.title("""Welcome to this toolbox for digital music analysis !
Prototype for the MusiScale research project, funded by the french National Research Agency""")
st.space(size="medium")


col1, col2 = st.columns(2, gap="xxsmall", vertical_alignment="center")
with col1:
    if st.button("Compute a formal diagram"):
        st.switch_page("pages/FormalDiagramApp.py")

with col2:
    if st.button("Compute a time-duration diagram"):
        st.switch_page("pages/TimeDurationDiagram.py")


st.space("medium")
st.text_area("About the tools presented here", height="content", value="""This project allows you to use the well known self-similarity matrix in order to compute new representations of the music, while keeping a signal-based approach. On the one hand, you can create what we call a "formal diagram", a representation which converts your self-similarity matrix into a musical visualization of your audio sample. The y-axis counts the materials instanciated in the audio when they are new and the x-axis is still the timeline.
This representation is a formal representation in the sense that it shows the organization of the materials into the musical time.

On the other hand, the project allows you to create what we can call a "time-duration diagram", a representation which computes the duration of the events on the y-axis, while keeping the time on the x-axis. This representation is obtained by making an affine transformation of the self-similarity matrix ; indeed, a triangulation of the coordinates. In such a representation, all coordinates represent a comparison between the two other points of the triangular event you can see in the matrix.
             
""")

st.space(size="medium")
st.text_area("Yes, but we need a similarity measure...", height="content", value="""Of course, we need a similarity measure in order to compute the self-similarity matrix ! For now, we just use one similairity measure : the concordance. Shortly, the concordance tells us how much energy is shared between two adjacent frames (or events) of your signal.
             
So we calculate the spectrum of the signal by the STFT and we build the matrix with the dot product of each pair of frames. Then we clip the similarity values between 0 and 1. So what we observe in the matrix is a measure of energy shared by the frames.
             
Later, we will propose other similarity measures, and even some combinations of similarity measures.""")

st.space("medium")
st.text("Note that this application is a prototype made in the framework of the MusiScale research project, funded by the french National Research Agency. The goal of this interdisciplinary research project is to propose a multi-scale representation of the musical structure and to formulate a theory behind this representation.")