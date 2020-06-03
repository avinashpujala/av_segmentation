
import os
import inOut
import ffmpeg
import youtube_dl

yt_url = r'https://www.youtube.com/watch?v=9gvalQOmymw'
ydl_opts = {'format': '22/18',
            'quiet': True,
            'ignoreerrors': True,
            'no_warnin'
            'gs': True}
fileName = '/home/avi/Documents/blah.mp4'
startTime=1
endTime=3
vcodec = 'vp9' # ('libx264', 'rawvideo', 'vp9')
acodec = 'libvorbis' #('aac', 'pcm', 'libvorbis')

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    download_url = ydl.extract_info(url=yt_url, download=False)['url']

ffin = ffmpeg.input(download_url, ss=startTime, to=endTime)
ffout=ffin.output(fileName, format='mp4', r=25, vcodec=vcodec,
                  crf=18, preset='veryfast', pix_fmt='yuv420p', acodec=acodec,
                  audio_bitrate=128000, strict='experimental').global_args('-y')
# ffout=ffin.output(fileName, format='mp4')
# ffout=ffin.output(fileName, format='mp4').global_args('-y')
ffout = ffout.global_args('-loglevel', 'error')
# ffout.run()