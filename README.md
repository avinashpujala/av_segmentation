# av_segmentation
Audio segmentation in videos using visual information from regions of interest. 

It downloads only the relevant part of the video and saves it in a mp4 container.

Contains scripts for downloading AVSpeech dataset, 720p/360p videos with 25fps and audio at 44.1kHz. 
This part of the code has been adapted from Nabarun Goswami's code at

https://github.com/naba89/AVSpeechDownloader

Files names are <yt_id>\_<start_time>\_<end_time>.mp4

Assumptions/Limitations: 
  - `avspeech_train.csv` and `avspeech_test.csv` are in the same directory as the download.py script.
  - creates the output folder in the currect directory based on the train/test set. For now you have to change the directories in the code if required.
  - the script creates a file called `badfiles_train.txt` which lists the youtube id's of the deleted/private videos which are no longer available for download.
  
Usage:
  ```
  inOut.download_av_speech.py train
  ```
Replace train with test if you want to download the test set.

Dependencies:
```
  conda install -c conda-forge ffmpeg
  conda install -c conda-forge youtube-dl
  pip install ffmpeg-python
```
