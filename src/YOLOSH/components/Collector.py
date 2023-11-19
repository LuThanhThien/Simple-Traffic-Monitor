import os
import time
from tqdm import tqdm
import math
from components.utils import yaml_to_dict
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS

class Extractor:
    def __init__(self, opt) -> None:
        # validate the input arguments
        print("Validating input arguments...")
        self.validate_opt(opt)
        self.opt = opt
        time.sleep(1)
        print("Input arguments passed validation.\n")

        print(f"Validating yaml file {opt.dataExtract}")
        self.extractor_yaml_arg = ['path', 'videos', 'urls', 'output']
        self.videos, self.urls, self.output = self.validate_yaml(opt.dataExtract)
        time.sleep(1)
        print("Yaml file passed validation.\n")

        self.maxImages = opt.maxImages
        self.imgExt = opt.imgExt
        self.efps = opt.efps

    
    def validate_opt(self, opt):
        # validate the input arguments
        if not os.path.exists(opt.dataExtract):
            raise FileNotFoundError(f"Yaml file {self.opt.dataExtract} does not exist.")
        if opt.maxImages <= 0:
            raise ValueError("maxImages must be greater than 0.")
        if opt.efps < -1 or opt.efps == 0:
            raise ValueError("efps must be greater than 0 or -1 for automatic setting.")
        if opt.imgExt not in IMG_FORMATS:
            raise ValueError("imgExt must be in ", IMG_FORMATS)
    

    def validate_yaml(self, yaml_path):
        data = yaml_to_dict(yaml_path)
        
        if len(data.keys()) == 0:
            raise KeyError("Yaml file is empty. Please check the yaml file.")

        # output validation
        if 'output' not in data.keys():
            raise KeyError('No output path specified in yaml file.')
        output = data['output']

        videos = []
        urls = []

        # videos validation
        if 'videos' not in data.keys() or data['videos'] == []:
            print('WARNING: No videos specified in yaml file.')
        elif 'path' not in data.keys():
            raise KeyError('No path specified in yaml file.')
        else:
            if not os.path.exists(data['path']):
                raise FileNotFoundError(f"Path {data['path']} does not exist.")
            path = data['path']
            names = data['videos']
            # merge path and names
            for name in names:
                if os.path.exists(os.path.join(path, name)):
                    videos.append(os.path.join(path, name))
                else:
                    print(f"WARNING: {name} does not exist in {path}.")

            if len(videos) == 0:
                raise FileNotFoundError(f"No videos found! Please check the path and names in yaml file.")

        if 'urls' in data.keys():
            urls = data['urls']
        else: 
            print('WARNING: No urls specified in yaml file.')
        
        if len(data.keys()) > len(self.extractor_yaml_arg):
            print(f"""WARNING: Yaml file contains more than {len(self.extractor_yaml_arg)} keys. Extractor class keys are {self.extractor_yaml_arg}. 
                  Some keys not recognized in this class will be automatically excluded.""")
            
        return videos, urls, output


    def extract(self):
        import pafy
        print("Output path: ", self.output)
        print("Max number of extracted images: ", self.maxImages)
        print("Frame per Second: ", self.efps)
        print("Paths: ", self.videos)
        print("URLs: \n", self.urls)
        
        # Create directories for images and labels
        if self.videos != []:
            for video in self.videos:
                print('Extracting images from ', video)
                base = os.path.basename(video).split('.')[0]
                os.makedirs(os.path.join(self.output, base), exist_ok=True)
                self.extract_from_video(video, method='video')
                print('\n')
        if self.urls != [] and self.urls != None:
            for url in self.urls:
                print('Extracting images from ', url)
                video = pafy.new(url)
                base = video.title
                os.makedirs(os.path.join(self.output, base), exist_ok=True)
                self.extract_from_video(video, method='youtube')
                print('\n')
    

    def video_params(self, cap):
        import cv2
        # get video parameters
        length = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        toltal_time = int(length / cap_fps)

        if self.efps > cap_fps:
            print(f"EFPS = {self.efps} is too high. Setting EFPS to ", cap_fps)
            self.efps = cap_fps
        elif self.efps == -1:
            print("Automatic Setting EFPS to ", cap_fps)
            self.efps = max(1, 30-cap_fps)

        time.sleep(1)

        if self.efps > 10:
            print(f"""WARNING: EFPS = {self.efps} is too high. It may cause memory error and extracting images with same content.
                  This can result in a poor training dataset.""")
            time.sleep(1)

        real_length = int(toltal_time * self.efps)
        frame_ratio = math.floor(cap_fps / self.efps)
        total_images = min(real_length, self.maxImages)

        return cap_fps, frame_ratio, total_images


    def extract_from_video(self, video, method='video'):
        import cv2

        # load video
        if method == 'youtube':
            print("Loading video from YOUTUBE...")
            best = video.getbest(preftype="mp4")
            base = video.title
            cap = cv2.VideoCapture(best.url)
        elif method == 'video':
            base = os.path.basename(video).split('.')[0]
            cap = cv2.VideoCapture(video)

        # get video parameters
        cap_fps, frame_ratio, total_images = self.video_params(cap)
        print(f"Video FPS: {cap_fps}; extracted FPS: {self.efps}")
        print("Total saved images: ", total_images)
        print("Frame ratio: ", frame_ratio)

        frame_count = 1
        img_count = 1
        batch_count = 1

        # progress bar with tqdm
        pbar = tqdm(total=total_images, desc=f'Extracting images')
        file_log = tqdm(total=0, position=1, bar_format='{desc}')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # save image
                if (frame_count % frame_ratio == 0) and (batch_count <= self.efps):  
                    file_log.set_description_str(f'Saving image {img_count}th: {base}-{frame_count}{self.imgExt} ')

                    # pass if file already exists
                    if os.path.exists(os.path.join(self.output, base, f'{base}-{frame_count}{self.imgExt}')):
                        file_log.set_description_str(f'{base}-{frame_count}{self.imgExt} already exists.')
                        img_count += 1
                        batch_count += 1
                        pbar.update(1)
                        time.sleep(0.1)
                        pass
                    else:
                        cv2.imwrite(os.path.join(self.output, base, f'{base}-{frame_count}{self.imgExt}'), frame)
                        img_count += 1
                        batch_count += 1
                        pbar.update(1)
                        time.sleep(0.1)

                # update frame count and batch count
                if batch_count == self.efps + 1 and frame_count % cap_fps == 0:
                    batch_count = 1
                frame_count += 1
                    
                # break if max number of images is reached
                if img_count > self.maxImages:
                    break
            else:
                break

        file_log.set_description_str(f'Finished extracting images {video}.')
        pbar.close()


    def main(self):
        self.extract()



from pytube import YouTube
from tqdm import tqdm

class Downloader:
    def __init__(self, opt):
        # validate the input arguments
        print("Validating input arguments...")
        self.validate_opt(opt)
        self.opt = opt
        self.rename = opt.rename
        time.sleep(1)
        print("Input arguments passed validation.\n")
        
        self.yaml_path = opt.dataDownload
        print(f"Validating yaml file {opt.dataDownload}")
        self.urls, self.output = self.validate_yaml(opt.dataDownload)
        time.sleep(1)
        print("Yaml file passed validation.\n")


    def validate_opt(self, opt):
        if not os.path.exists(opt.dataDownload):
            raise FileNotFoundError(f"Yaml file {opt.dataDownload} does not exist.")

    
    def validate_yaml(self, yaml_path):
        data = yaml_to_dict(yaml_path)

        if len(data.keys()) == 0:
            raise KeyError("Yaml file is empty. Please check the yaml file.")
        if 'output' not in data.keys():
            raise KeyError('No output path specified in yaml file.')
        if 'urls' not in data.keys():
            raise KeyError('No urls specified in yaml file.')
        if len(data.keys()) > 2:
            print(f"""WARNING: Yaml file contains more than {len(data.keys())} keys. Extractor class keys are {data.keys()}. 
                  Some keys not recognized in this class will be automatically excluded.""")
        
        output = data['output']
        urls = data['urls']

        return urls, output
    

    def download(self):
        print('Output path: ', self.output)
        print('URLs: ', self.urls)
        print('Rename: ', self.rename)

        if self.rename:
            print('''WARNING: In rename mode, the downloaded files will be renamed as video-1, video-2, etc.
                  If you have files downloaded from the same URLs in the output directory, they will be ignored.
                  This can lead to DUPLICATE files in the output directory. Please check the URLs and output directory in yaml file.''')
            time.sleep(5)

        file_log = tqdm(total=0, position=1, bar_format='{desc}')
        pbar = tqdm(total=len(self.urls), desc="Downloading video")

        fails = []
        for url in self.urls:
            yt = YouTube(url)
            vid_name = yt.title
            video = yt.streams.filter(adaptive=True).first()
            file_log.set_description_str(f"Downloading video {vid_name} with resolution {video.resolution}")
            try:
                if self.rename:
                    filename=self.rename_file() + '.' + video.mime_type.split('/')[-1]
                    video.download(self.output, filename=filename)
                else:
                    video.download(self.output)
                pbar.update(1)
                time.sleep(0.1)
            except:
                file_log.set_description_str(f"Failed to download video {vid_name}")
                fails.append(vid_name)
                pbar.update(1)
                time.sleep(0.1)
                pass 
        
        file_log.set_description_str(f"Finished downloading videos.")
        pbar.close()

        if len(fails) == 0:
            pass
        elif len(fails) == len(self.urls):
            print("Failed to download all videos.")
        else:
            print(f'Failed to download {len(fails)} videos: ', fails)


    def rename_file(self):
        max_index = -1
        for file in os.listdir(self.output):
            if file.startswith('video-') and os.path.isfile(os.path.join(self.output, file)) and file.endswith(tuple(VID_FORMATS)):
                file_name = file.split('.')[0]
                index = int(file_name.split('-')[1])
                if index > max_index:
                    max_index = index
        return 'video-' + str(max_index + 1)

    def main(self):
        self.download()




# def download_audio(video_url):
#     video = YouTube(video_url)
#     audio = video.streams.filter(only_audio = True).first()

#     try:
#         audio.download()
#         print("audio was downloaded successfully")
#     except:
#         print("Failed to download audio")

