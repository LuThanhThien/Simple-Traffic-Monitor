YOLOSH - a YOLO Simple Helper 

# problem with youtube_dl:
lib\site-packages\youtube_dl\extractor\common.py", line 534

'uploader_id': self._search_regex(r'/(?:channel|user)/([^/?&#]+)', owner_profile_url, 'uploader id') if owner_profile_url else None,

-> 'uploader_id': self._search_regex(r'/(?:channel/|user/|@)([^/?&#]+)', owner_profile_url, 'uploader id', default=None),

lib\site-packages\pafy\backend_youtube_dl.py", line 53

comment those lines:

self._likes = self._ydl_info['like_count']

self._dislikes = self._ydl_info['dislike_count']
