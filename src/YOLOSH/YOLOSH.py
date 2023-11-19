from components.Collector import Extractor, Downloader
from components.Ingestor import Ingestor
from components.Trainer import Trainer, Predictor
import os

def parse_opt():
        import argparse

        def list_str(values):
            return values.split(',')

        parser = argparse.ArgumentParser()

        parser.add_argument('-c', '--className', type=str, help='Specify which class to use')

        yolosh_dir = os.path.dirname(__file__)
        extract_yaml = os.path.join(yolosh_dir, 'yaml//collect//extract.yaml')
        download_yaml = os.path.join(yolosh_dir, 'yaml//collect//download.yaml')
        ingest_yaml = os.path.join(yolosh_dir, 'yaml//ingest//ingest.yaml')
        train_yaml = os.path.join(yolosh_dir, 'yaml//train//train.yaml')
        predict_yaml = os.path.join(yolosh_dir, 'yaml//train//predict.yaml')

        # Collector settings
        parser.add_argument('--dataExtract', type=argparse.FileType('r'), default=extract_yaml, help='Yaml file for extract data')
        parser.add_argument('--imgExt', type=str, default='.jpg', help='Output image extensions')
        parser.add_argument('--maxImages', type=int, default=100, help='Maximum number of images to collect per video')
        parser.add_argument('--efps', type=int, default=-1, help='Number of extracted frame per second')
        parser.add_argument('--dataDownload', type=argparse.FileType('r'), default=download_yaml, help='Yaml file for download data')
        parser.add_argument('--rename', action='store_true', help='Rename the downloaded files')

        # Ingestor settings
        parser.add_argument('--dataIngest', type=argparse.FileType('r'), default=ingest_yaml, help='Yaml file for ingest data') 
        parser.add_argument('--ratio', type=float, default=0.8, help='Spliting ratio')
        parser.add_argument('--mode', type=str, choices=['split', 'merge'] default='split', help='Spliting mode: split or merge')
        parser.add_argument('--method', type=str, choices=['copy', 'cut'], default='copy', help='Moving type: cut or copy')
        parser.add_argument('--ignoreBlank', action='store_true', help='Ignore blank label files')

        # Trainer settings
        parser.add_argument('--weightsTrain', type=argparse.FileType('r'), default='yolov8l.pt', help='Path of YOLO model weights')
        parser.add_argument('--dataTrain', type=argparse.FileType('r'), default=train_yaml, help='Yaml file for train data')
        parser.add_argument('--batch', type=int, default=-1, help='Number of batch size')
        parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
        parser.add_argument('--imgsz', type=int, default=640, help='Image size')
        parser.add_argument('--workers', type=int, default=1, help='Number of loader workers')
        parser.add_argument('--resume', action='store_true', help='Resume the training process by a pretrained weights')
        parser.add_argument('--project', type=str, default='runs', help='Path of project') 
        parser.add_argument('--name', type=str, default='detect', help='Name of project')   
        parser.add_argument('--shutdown', action='store_true', help='WARNING: Shut down your local machine when finish traing')
        

        # Predictor settings
        parser.add_argument('--weightsPred', type=argparse.FileType('r'), default='yolov8l.pt', help='Path of YOLO model weights')
        parser.add_argument('--dataPred', type=argparse.FileType('r'), default=predict_yaml, help='Yaml file for data')
        parser.add_argument('--includeImg', action='store_true', help='To include image in prediction')
        parser.add_argument('--includeVid', action='store_true', help='To include video in prediction')
        parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
        parser.add_argument('--save', action='store_true', help='To save image with result')
        parser.add_argument('--save_train', action='store_true', help='To save the result in YOLO trainable .txt file')
        parser.add_argument('--annotate', action='store_true', help='To open annotate tool labelImg or not after prediction')

        return parser.parse_args()


if __name__ == "__main__":

    opt = parse_opt()
    class_name_str = opt.className

    print("<<Starting>>",class_name_str)

    class_dict = {
         'Extractor': Extractor, 
         'Downloader': Downloader,
         'Ingestor': Ingestor, 
         'Trainer': Trainer, 
         'Predictor': Predictor,
         }

    if class_name_str not in class_dict.keys():
        raise ValueError(f"Class {class_name_str} not found.")
    else:
        class_name = class_dict[class_name_str](opt)
        print(f"<<Summary {class_name}>>")
        print(class_name.opt)
        class_name.main()

