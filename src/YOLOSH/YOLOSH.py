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
        train_yaml = os.path.join(yolosh_dir, 'yaml//train//detect.yaml')
        predict_yaml = os.path.join(yolosh_dir, 'yaml//train//predict.yaml')

        # Collector settings
        parser.add_argument('--dataExtract', type=str, default=extract_yaml, help='Yaml file for extract data')
        parser.add_argument('--imgExt', type=str, default='.jpg', help='Output image extensions')
        parser.add_argument('--maxImages', type=int, default=100, help='Maximum number of images to collect per video')
        parser.add_argument('--efps', type=int, default=-1, help='Number of extracted frame per second')
        parser.add_argument('--dataDownload', type=str, default=download_yaml, help='Yaml file for download data')
        parser.add_argument('--rename', action='store_true', help='Rename the downloaded files')

        # Ingestor settings
        parser.add_argument('--dataIngest', type=str, default=ingest_yaml, help='Yaml file for ingest data') 
        parser.add_argument('--ratio', type=float, default=0.8, help='Spliting ratio')
        parser.add_argument('--mode', type=str, choices=['split', 'merge', 'relabel'], default='split', help='Ingesting mode: split, merge or relabel')
        parser.add_argument('--method', type=str, choices=['copy', 'cut'], default='copy', help='Moving type: cut or copy')
        parser.add_argument('--ignoreBlank', action='store_true', help='Ignore blank label files')

        # Trainer settings
        parser.add_argument('--weightsTrain', type=str, default='yolov8l.pt', help='Path of YOLO model weights')
        parser.add_argument('--dataTrain', type=str, default=train_yaml, help='Yaml file for train data')
        parser.add_argument('--batch', type=int, default=-1, help='Number of batch size')
        parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
        parser.add_argument('--imgsz', type=int, default=640, help='Image size')
        parser.add_argument('--workers', type=int, default=1, help='Number of loader workers')
        parser.add_argument('--resume', action='store_true', help='Resume the training process by a pretrained weights')
        parser.add_argument('--project', type=str, default='runs', help='Path of project') 
        parser.add_argument('--name', type=str, default='detect', help='Name of project')   
        parser.add_argument('--shutdown', action='store_true', help='WARNING: Shut down your local machine when finish traing')
        

        # Predictor settings
        parser.add_argument('--weightsPred', type=str, default='yolov8l.pt', help='Path of YOLO model weights')
        parser.add_argument('--dataPred', type=str, default=predict_yaml, help='Yaml file for data')
        parser.add_argument('--includeImg', action='store_true', help='To include image in prediction')
        parser.add_argument('--includeVid', action='store_true', help='To include video in prediction')
        parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
        parser.add_argument('--save', action='store_true', help='To save image with result')
        parser.add_argument('--save_txt', action='store_true', help='To save the result in YOLO trainable .txt file')
        parser.add_argument('--annotate', action='store_true', help='To open annotate tool labelImg or not after prediction')
        parser.add_argument('--map_cls', action='store_true', help='To open annotate tool labelImg or not after prediction')
        parser.add_argument('--semi', action='store_true', help='Semi-prediction from available labels')


        return parser.parse_args()


def main2():
    data_path = r'C:\Users\USER\Projects\20231019-traffic-management\src\assets\data\ocr_data'
    test = os.path.join(data_path, 'test')
    train = os.path.join(data_path, 'train')
    valid = os.path.join(data_path, 'valid')
    for path in [test, train, valid]:
        for folder in [os.path.join(path,'images'), os.path.join(path,'labels')]:
            for file in os.listdir(folder):
                if file.split('.')[0].endswith('bitwise') or file.split('.')[0].endswith('blur'):
                    file_path = os.path.join(folder, file)
                    os.remove(file_path)
          
if __name__ == "__main__":
    # main2()
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

