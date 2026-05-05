########################### COMPUTE A FORMAL DIAGRAM FROM A SELF-SIMILARITY MATRIX ##########################

import streamlit as st
import numpy as np 
import pandas as pd 
import datetime
import librosa
import soundfile as sf
import IPython.display as ipd
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6


def compute_ssm_contrast(SSM, threshold_selection):
    Threshold = threshold_selection ### define a similarity threshold, default = 0.95
    n_materials = SSM.shape[1]
    S_contrast = np.zeros((n_materials, n_materials))
    for i in range(n_materials):
        for j in range(i, n_materials):
            if i > j:
                S_contrast[i,j] = 0
            else:
                if S[i,j] >= Threshold:
                    S_contrast[i,j] = 1
                else:
                    S_contrast[i,j] = 0
    return S_contrast


def compute_formal_diagram(contrasting_ssm):
    n_materials2 = contrasting_ssm.shape[1]
    FormalDiagram = np.zeros((n_materials2, n_materials2))
    k = 0
    for j in range(n_materials2):
        for i in range(n_materials2 - 1):
            FormalDiagram[i,j] = contrasting_ssm[i,j]
    for j in range(n_materials2):
        for i in range(0,j+1):                
            if ((i < (j-k)) and (FormalDiagram[i,j] == 1)) or ((i == j-k) and (FormalDiagram[i,j] == 0)):
                jj = j
                ii = i
                for ii in range(j-k, (n_materials2 - 1)):
                    for jj in range((j), (n_materials2)):                    
                        FormalDiagram[ii,jj] = FormalDiagram[ii+1,jj]
                k += 1
    return FormalDiagram


#### removing uncessessary matrix lines (all lines with 0) in order to reshape the plot with correct dimensions
def  m_reshape(m):
    mask = np.any(m, axis=1) # test if there is at least 1 non-zero element for each line and keep only the concerned lines (return boolean vector)
    return m[mask] #apply the mask to the matrix, so delete all lines with only zero elements (this is indeed a filter)


def display_ssm(matrix, hop_length, start, duration, tick_spacing_selection):
    # Convert frame indices to time
    n_frames = matrix.shape[1]
    frame_indices = np.arange(n_frames)
    times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hop_length)
    times = times + start
    
    duration = times[-1] # time of the last frame of the array, so approximatively the total duration of the signal

    #color map
    cmap = libfmp.b.compressed_gray_cmap(alpha=-500)

    fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axes objects (square shape)
    im_matrix = ax.imshow(matrix, aspect='equal', origin='upper', cmap=cmap, extent=[times[0], times[-1], times[-1], times[0]]) # cmap parameter as defined above
    # with the extent parameter, we map the matrix coordinates to real time in sec when displaying the image, for coordinates (x,y) (with y from the top)

    plt.colorbar(im_matrix, ax=ax, label='Similarity', fraction=0.04, pad=0.1) #set the colobar : k=width ; pad=distance between axes and colorbar
    #ax.set_title('Self-Similarity Matrix using Concordance Measure')

    ax.spines['bottom'].set_visible(False) # Delete the redundant x-axis at the bottom of the matrix
    ax.yaxis.set_visible(True) # display y-axis and ticks, labels etc.
    ax.spines['left'].set_visible(True)
    
    # Move x-axis to the top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    #delete the grid under the diagonal : this method clips the grid using a triangular patch from matplotib.patches method
    n = matrix.shape[0]
    triangle = patches.Polygon(     # the coordinates are (0,0) for top left, (0,n) for bottom left and (n,n) for bottom right : we simply draw a white polygon around these values (cf cmap)
    [[0, 0], [0, n], [n, n]], 
    closed=True,
    facecolor='white',
    edgecolor='none',
    zorder=3)
    ax.add_patch(triangle)
    
    # ticks spacing
    ticks = np.arange(start, start + duration - start, tick_spacing_selection) #here we change the tick spacing (in second) (default value : all 5 sec)
    tick_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in ticks] 

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel('Time (min:sec)')
    ax.set_ylabel('Time (min:sec)')

    ax.grid(False, axis='x')
    ax.grid(False, axis='y')

    return st.pyplot(fig, width="content")

def display_formal_diagram(diagram, hop_length_selection, start, duration, tick_spacing_selection):
    # Convert frame indices to time
    n_framesDiagram = diagram.shape[1]
    frame_indices_diagram = np.arange(n_framesDiagram)
    times = librosa.frames_to_time(frame_indices_diagram, sr=sr, hop_length=hop_length_selection)
    times = times + start
    
    duration = times[-1] # time of the last frame of the array, so approximatively the total duration of the signal

    #color map
    cmap = libfmp.b.compressed_gray_cmap(alpha=-500)

    fig, ax = plt.subplots()  # Create figure and axes objects (square shape)
    im_diagram = ax.imshow(m_reshape(diagram), aspect=0.3, origin='upper', cmap=cmap, extent=[times[0], times[-1], m_reshape(diagram).shape[0]-1, 0]) # cmap parameter as defined above
    # with the extent parameter, we map the matrix coordinates to real time in sec when displaying the image, for coordinates (x,y) (with y from the top)
    # in the case of the formal diagram, we keep the (m_reshape(FormalDiagram).shape[0]-1) number of lines in our plot

    ax.spines['bottom'].set_visible(False) # Delete the redundant x-axis at the bottom of the matrix
    ax.yaxis.set_visible(True) # display y-axis and ticks, labels etc.
    ax.spines['left'].set_visible(True)
    
    # Move x-axis to the top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # ticks spacing
    xticks = np.arange(start, start + duration - start, tick_spacing_selection) #here we change the tick spacing (in second) (default value : all 5 sec)
    tick_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in xticks] 

    ax.set_xticks(xticks)
    ax.set_xticklabels(tick_labels)

    #resize yticks
    ax.set_yticks(range(m_reshape(diagram).shape[0]))
    #ax.set_yticklabels(tick_labels)

    ax.set_xlabel('Time (min:sec)')
    ax.set_ylabel('Materials')

    ax.grid(False, axis='x')
    ax.grid(False, axis='y')

    return st.pyplot(fig, width="content")




#================================= main program ===================================#

# title of the page
st.title("Formal Diagram")
st.space("small")
st.text("""This page allows you to compute a formal diagram from a self-similarity matrix. 
        
First, select an audio file with the starting time and the duration you want (in sec) (example : start = 0 and duration = 10). The program display the waveform of your selection. Then, select the basic parameters of the FFT (hop_length and window) in order to plot it.

Now you can plot a self-similarity matrix based on the concordance as similairty measure : the concordance tells how much energy is shared between two adjacent frames. For a better display, choose the appropriate ticks spacing (depends on the duration of your audio sample).

Finally, you can compute a formal diagram, a new representation which converts your self-similarity matrix into a musical visualization of your audio sample. The y-axis counts the materials instanciated in the audio when they are new and the x-axis is still the timeline.""")

st.space("small")
# select an audio file
# if audio file selected, make computations
st.subheader("Audio sample")
audio_file = st.file_uploader("Choose an audio file", type="audio")
if audio_file:
    st.write("filename:", audio_file.name)
    with NamedTemporaryFile() as f:
        f.write(audio_file.read())
        wave, sr = librosa.load(f.name, sr=None)
    
        times = pd.Index(np.array(range(len(wave))) / sr, name="time(sec)")
        ts = pd.Series(wave, index=times, name=str(audio_file))
    #st.write(ts)

        start = st.number_input("Enter the starting time", min_value=0.0, max_value=times.to_list()[-1])
        duration = st.number_input("Enter the duration", min_value=0.1, max_value=times.to_list()[-1]-start)
        end = start + duration

        wave2, sr = librosa.load(f.name, sr=sr, offset=start, duration=duration)

    st.html(ipd.Audio(wave2, rate=sr)._repr_html_())

    #display waweform
    fig, ax = plt.subplots()
    librosa.display.waveshow(wave2, sr=sr, max_points=20000, axis='m')
    st.pyplot(fig, width="content")

    print(start, duration, end)

    st.space("medium")
    # fft spectrum
    st.subheader("Spectrum (short-term fourier transform)")
    hop_length_selection = st.selectbox("hop length for FFT", [512, 1024, 2048], index=0) #default = 512
    window_selection = st.selectbox("window for FFT", ["blackmanharris","blackman","hann", "hamming"], index=3) #default = hamming
    Xfft = np.abs(librosa.stft(wave2, hop_length=hop_length_selection, window=window_selection))
    Xfft_db = librosa.amplitude_to_db(Xfft, ref=np.max)
    fig, ax = plt.subplots()
    librosa.display.specshow(Xfft_db, sr=sr, x_axis="time", y_axis="log", cmap='grey')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(fig, width="content")


    st.space("medium")
    #self-similarity matrix
    st.subheader("Self-similarity matrix with concordance measure")
    n_frames = Xfft.shape[1] ### frame ==> event
    S = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        for j in range(i, n_frames):
            S[i, j] = np.dot(Xfft[:, i].T, Xfft[:, j])

    # Ensure values are in proper range [0, 1] for similarity, with a percentil-based normalization 
    vmin, vmax = np.percentile(S, [5, 95])
    S = np.clip((S - vmin) / (vmax - vmin), 0, 1)
    #contrast enhancement
    S = (np.clip(S, 0.5, 1) - 0.5)*2

    # display the matrix
    tick_spacing_selection = st.selectbox("Select a tick spacing (in sec) for the display of the self-similarity matrix", 
                                          [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 60], index=0)
    display_ssm(S, hop_length_selection, start, duration, tick_spacing_selection)

    
    st.space("medium")

    #compute contrasting self-similarity matrix
    st.subheader("Self-similarity matrix with a similarity threshold")
    threshold_selection = st.number_input("""Select a similarity threshold in order to compute a contrasting self-similarity matrix.
                                       This new matrix will be a basis for the formal diagram. Choose a value between 0 and 1. 
                                          """, min_value=0.0, max_value=1.0, value=0.95) # default = 0.95
    ssm_with_threshold = compute_ssm_contrast(S, threshold_selection)
    display_ssm(ssm_with_threshold, hop_length_selection, start, duration, tick_spacing_selection)
    

    # compute the formal diagram
    st.space("medium")
    st.text("Now here is the formal diagram !")
    st.space("small")
    st.subheader("Formal Diagram with a concordance measure")
    formal_diagram = compute_formal_diagram(ssm_with_threshold)
    display_formal_diagram(formal_diagram, hop_length_selection, start, duration, tick_spacing_selection)
