
import yt_dlp
import numpy as np
import time
import datetime

# datetime
now = datetime.datetime.now()
time.sleep(0.3)
then = datetime.datetime.now()
diff = then - now
print("Datetime measurements")
print(diff, diff.total_seconds(), type(diff.total_seconds()))

# # time
# now = time.time()
# time.sleep(1)
# then = time.time()
# diff = then - now
# print("Time measurements")
# print(diff)

# test = [1,2,3,4,5,95]
# test2 = [94,95,96]

# test3 = any(item in test for item in test2)
# print(test3)
# import sys
# sys.exit()

# LIVESTREAM_IDS = list(range(91, 97))

# class VideoStream:
#     url: str = None
#     resolution: str = None
#     height: int = 0
#     width: int = 0

#     def __init__(self, video_format):
#         self.url = video_format['url']
#         self.resolution = video_format['format_note']
#         self.height = video_format['height']
#         self.width = video_format['width']

#     def __str__(self):
#         return f'{self.resolution} ({self.height}x{self.width}): {self.url}'

# url = "https://www.youtube.com/watch?v=YLSELFy-iHQ" # live
# # url = "https://www.youtube.com/watch?v=7cbRcbXwezA" # modzy
# ydl_opts = {}
# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     info = ydl.extract_info(url, download=False)
#     # filter to liavestream ids
#     # info_livestreams = [format for format in info if format['format_id'] in ]
#     # print(info['formats'])
#     for format in info['formats']:
#         # print(format['vcodec'], format['format_note'] if 'format_note' in format else None)
#         print('format id: ', format['format_id'])
#         print('url: ', format['url'] if 'url' in format else None)
#         print('resolution: ', format['resolution'])
#         print("fps: ", format['fps'] if 'fps' in format else None)
#         print('quality: ', format['quality'])
#         print(format.keys())
#         print("\n\n\n")
#     print(len(info['formats']))

    # streams = [VideoStream(format)
    #             for format in info['formats'][::-1]
    #             if format['vcodec'] != 'none' and 'format_note' in format]
    # print(streams)
    # _, unique_indices = np.unique(np.array([stream.resolution
    #                                         for stream in streams]), return_index=True)
    # print(unique_indices)
    # streams = [streams[index] for index in np.sort(unique_indices)]
    # resolutions = np.array([stream.resolution for stream in streams])
    # print(streams, resolutions)


