import streamlit as st
import os


def select_file(path_folder='.', ext='mp4'):
	fileNames = os.listdir(path_folder)
	fileNames = [f for f in fileNames if f'.{ext}' in f]
	selectedFile = st.selectbox('Select movie file: ', fileNames)
	return os.path.join(path_folder, selectedFile)


st.markdown('# Sound Source Isolation Demo\n ## *Avinash Pujala*')
st.markdown('\n\n')

fileName = select_file()
fileName_only = os.path.split(fileName)[-1]

st.write(f'You selected: \t\t\t \"{fileName_only} \"')

#%% Video
st.header(f'Original video: {fileName_only}')
st.video('Girl_singing_and_drumming.mp4', format='video/mp4')


#%% Image x Mask
st.header('Mask to extract drum sounds')
st.image('imgxmask.png')


#%% Drum sounds only
st.markdown('# Drum sounds returned by the network')
st.audio('drums_only.wav')

