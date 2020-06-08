# av_segmentation
Audio segmentation in videos using visual information from regions of interest. 

Contains scripts for downloading AVSpeech dataset, 720p/360p videos with 25fps and audio at 44.1kHz. 
This part of the code has been adapted from Nabarun Goswami's code at
https://github.com/naba89/AVSpeechDownloader

Get this as follows:
Assumptions/Limitations: 
  - the script creates a file called `badfiles_train.txt` which lists the youtube id's of the deleted/private videos which are no longer       available for download.  

Usage:
  ```
  inOut.download_av_speech.py train
  ```
Dependencies:
```
  conda install -c conda-forge ffmpeg
  conda install -c conda-forge youtube-dl
  pip install ffmpeg-python
```
In addition requires ```mediaio``` from https://github.com/avivga/mediaio.git
Get this as follows:
```
git clone --single-branch -branch --depth 1  https://github.com/avivga/mediaio.git
cd mediaio
python setup.py --user
```

