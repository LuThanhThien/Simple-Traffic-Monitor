from ultralytics import YOLO
import torch 
import time
from tqdm import tqdm
import os
import ctypes
import glob
from src.logger import logging
from src.util.utils import loggingInfo
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from components.utils import yaml_to_dict

class Trainer:
    def __init__(self, opt) -> None:
        self.opt = opt

    def train(self):
        try:
            start = time.time()
            model = YOLO(self.opt.weightsTrain)
            loggingInfo("Training start >>>")
            result = model.train(data=self.opt.dataTrain, batch=self.opt.batch, epochs=self.opt.epochs, resume=self.opt.resume,
                                workers=self.opt.workers, imgsz=self.opt.imgsz, project=self.opt.project, name=self.opt.name,
                                degrees=60, flipud=0, fliplr=0, shear=30)
            end = time.time()
            loggingInfo('Finish training in ', end-start, ' seconds')

            if self.opt.shutdown:
                loggingInfo('WARNING: Finished training. Do you still want to shut down? After 5 minutes shut down will be automatically implemented')
                ctypes.windll.user32.MessageBoxW(0, "Finished training. Do you still want to shut down? After 5 minutes shut down will be automatically implemented", "Warning", 1)
                # Wait for 5 minutes
                time.sleep(300) 
                os.system('shutdown -s')
        except Exception as e:
            raise e

    def main(self):
        self.train()

class Predictor:
    def __init__(self, opt) -> None:
        print("Validating input arguments...")
        self.validate_opt(opt)
        time.sleep(1)
        print("Input arguments passed validation.\n")
        self.opt = opt
        self.weightsPred = opt.weightsPred
        self.model = YOLO(self.weightsPred)


        print(f"Validating yaml file {opt.dataPred}")
        self.pred_yaml_arg = ['input', 'output', 'classes']
        self.input, self.output, self.class_dict = self.validate_yaml(opt.dataPred)
        time.sleep(1)
        print("Yaml file passed validation.\n")
        
        self.includeImg = opt.includeImg
        self.includeVid = opt.includeVid
        self.ext = []
        self.ext.extend(IMG_FORMATS) if self.includeImg else None
        self.ext.extend(VID_FORMATS) if self.includeVid else None

        self.conf = opt.conf
        self.save = opt.save
        self.save_train = opt.save_train
        self.annotate = opt.annotate


    def validate_opt(self, opt):
        if opt.weightsPred is None:
            raise ValueError("weightsPred must be specified")
        # elif not os.path.exists(opt.weightsPred):
        #     raise ValueError(f"weightsPred {opt.weightsPred} does not exist.")
        if opt.includeImg is False and opt.includeVid is False:
            raise ValueError("At least one of includeImg or includeVid must be True")
        if not os.path.exists(opt.dataPred):
            raise ValueError(f"Yaml file {opt.dataPred} does not exist.")
        if opt.conf < 0 or opt.conf > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        

    def validate_yaml(self, yaml_path):
        data = yaml_to_dict(yaml_path)

        # check for yaml file
        if len(data.keys()) == 0:
            raise KeyError("Yaml file is empty. Please check the yaml file.")
        if len(data.keys()) < len(self.pred_yaml_arg):
            raise KeyError(f"Yaml file must contain {self.pred_yaml_arg} keys. Please check the yaml file.")
        
        # check input path
        if not os.path.exists(data['input']):
            raise FileNotFoundError(f"Input path {data['input']} does not exist.")
        
        # check output path
        if not os.path.exists(data['output']):
            print(f"Output path {data['output']} does not exist. Creating new directory...")
            os.makedirs(data['output'])
            time.sleep(1)

        # check for classes
        class_keys = []
        if data['classes'] is None or data['classes'] == []:
            print("WARNING: Classes is empty, automatic assigning all model classes.")
            data['classes'] = self.model.names
            class_keys = list(self.model.names.keys())
            time.sleep(1)
        else:
            exclude = []
            all_classes = list(self.model.names.values())
            for cls in data['classes']:
                if cls not in all_classes:
                    exclude.append(cls)
                    data['classes'].remove(cls)
                else:
                    class_keys.append(all_classes.index(cls))
        
        if len(exclude) > 0:
            print(f"WARNING: The following classes are not in the model will be excluded: {exclude}")
            time.sleep(1)

        input = data['input']
        output = data['output']
        classes = data['classes']

        return input, output, dict(zip(class_keys, classes))

    def find_media(self, path_list:list=[], root=None):        
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(tuple(self.ext)):
                    new_path = os.path.join(root, file)
                    path_list.append(new_path)
            for dir in dirs:
                path_list.extend(self.find_media(path_list, dir))
        return path_list


    def make_up_classes_txt(self):
        # saving classes.txt file
        if not os.path.exists(os.path.join(self.output, 'labels')):
            os.makedirs(os.path.join(self.output, 'labels'), exist_ok=True)
        
        classes_txt_path = os.path.join(self.output, 'labels', 'classes.txt')
        with open(classes_txt_path, 'w') as f:
            for cls in self.class_dict.values():
                f.write(cls+'\n')

        predefined_classes_path = 'labelImg\\data\\predefined_classes.txt'
        open(predefined_classes_path, 'w').close()
        with open(predefined_classes_path, 'w') as f:
            for cls in self.class_dict.values():
                f.write(cls+'\n')
        

    def predict(self):

        print('Input path: ', self.input)
        print('Output path: ', self.output)
        print('Weights path: ', self.weightsPred)
        print('Classes included: ', self.class_dict.values())
        print('Include image: ', self.includeImg)
        print('Include video: ', self.includeVid)
        print('Yaml file path: ', self.opt.dataPred)
        

        print("Finding media files and predict >>>")
        path_list = self.find_media(root=self.input)
        print('Found {} media files.'   .format(len(path_list)))

        project = os.path.dirname(self.output)
        name = os.path.basename(self.output)

        print('Making up classes txt files.')
        if self.save_train:
            self.make_up_classes_txt()
            
        path_list_tqdm = tqdm(path_list, position=0)
        for path in path_list_tqdm:
            path_list_tqdm.set_description_str(f"Processing {os.path.basename(path)}" )
            predictions = self.model.predict(
                path,
                save=self.save,
                save_txt=self.save_train,
                conf=self.conf,
                project=project,
                name=name,
                exist_ok=True,
                classes=list(self.class_dict.keys())
            )

        if self.annotate:
            print("Opening labelImg...")
            import os
            os.system(f"cd labelImg && python labelImg.py {self.input} {os.path.join(self.output, 'labels', 'classes.txt')} ")
        return

    def main(self):
        self.predict()

