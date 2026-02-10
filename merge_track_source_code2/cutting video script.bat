:: Change the source and destination video file paths
set source_video="J:\MITICA\GH010371_data2.MP4"
set destination_video="J:\MITICA\trim_data2.MP4"

:: Set the start and stop times of the new video (in seconds)
set start_time="0"
set stop_time="120"

:: Command to trim the beginning and end of your video. No changes are needed to this line.
"%programfiles%\VideoLAN\VLC\vlc.exe" %source_video% --start-time=%start_time% --stop-time=%stop_time% --sout "#gather:std{access=file,dst=%destination_video%}" --sout-keep vlc://quit