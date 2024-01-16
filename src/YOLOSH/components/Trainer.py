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

        self.map_dict = {}    
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
        self.save_txt = opt.save_txt
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
        exclude = {}
        if data['classes'] is None or data['classes'] == {}:
            print("WARNING: Classes is empty, automatic assigning all model classes.")
            data['classes'] = self.model.names
            time.sleep(1)
        else:
            all_classes = list(self.model.names.values())
            for key, cls in data['classes'].items():
                if cls not in all_classes:
                    exclude[key] = cls
        
        if len(exclude) > 0:
            for key, cls in exclude.items():
                data['classes'].pop(key)
            print(f"WARNING: The following classes are not in the model will be excluded: {list(exclude.values())}")
            time.sleep(1)

        if self.opt.map_cls:
            reverse_dict = {v: k for k, v in data['classes'].items()}
            for key, cls in self.model.names.items():
                if cls in reverse_dict.keys():
                    self.map_dict[key] = reverse_dict[cls]
            print(f"Mapping classes to {self.map_dict}")
            time.sleep(1)
        
        input = data['input']
        output = data['output']
        class_dict = data['classes']
        
        return input, output, class_dict

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
        with open(classes_txt_path, 'a') as f:
            for cls in self.class_dict.values():
                f.write(str(cls)+'\n')

        predefined_classes_path = 'labelImg\\data\\predefined_classes.txt'
        os.makedirs(os.path.dirname(predefined_classes_path), exist_ok=True)
        open(predefined_classes_path, 'w').close()
        with open(predefined_classes_path, 'w') as f:
            for cls in self.class_dict.values():
                f.write(str(cls)+'\n')
    
    def check_same_txt_path(self):
        # check if path is the same as output path
        if os.path.dirname(self.input) == self.output:
            return True
        else:
            return False
        

    def save_txt_file(self, path, predictions):
        # saving txt file
        if self.check_same_txt_path():
            raise ValueError("Input path and output path cannot be the same. This leading to overwriting the original txt file. Please change the output path.")
        txt_path = os.path.join(self.output, 'labels', os.path.basename(path).replace(os.path.splitext(path)[1], '.txt'))
        with open(txt_path, 'a') as f:
            for cls, box in zip(predictions.cls, predictions.xywhn):
                if int(cls) not in self.map_dict.keys():
                    continue
                pred_key = cls if not self.opt.map_cls else self.map_dict[int(cls)]
                f.write(f"{pred_key} {box[0]} {box[1]} {box[2]} {box[3]}\n")

    def save_semi_txt_file(self, path, txt_lines):
        if self.check_same_txt_path():
            raise ValueError("Input path and output path cannot be the same. This leading to overwriting the original txt file. Please change the output path.")
        # saving txt file, path is the image path
        txt_name = os.path.basename(path).replace(os.path.splitext(path)[1], '.txt')
        txt_path = os.path.join(self.output, 'labels', txt_name)
        self.writelines_without_duplicate(txt_path, txt_lines)

    @staticmethod
    def writelines_without_duplicate(txt_path, txt_lines):
        if os.path.exists(txt_path):
            # saving txt file, path is the image path
            with open(txt_path, 'r+') as f:
                real_lines = f.readlines()
            with open(txt_path, 'a+') as f:
                for line in txt_lines:
                    if line not in real_lines:
                        print('writing line: ', line)
                        f.write(line)
        else:
            with open(txt_path, 'w+') as f:
                f.writelines(txt_lines) 

    def predict(self):
        import os 
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

        
        if self.save_txt:
            print('Making up classes txt files.')
            self.make_up_classes_txt()

        with tqdm(total=len(path_list), position=0, leave=True) as pbar:
            for path in path_list:
                pbar.set_description_str(f"Processing {os.path.basename(path)}" )
                predictions = self.model.predict(
                    path,
                    save=self.save,
                    save_txt=None,
                    conf=self.conf,
                    project=project,
                    name=name,
                    exist_ok=True,
                    classes=list(self.class_dict.keys())
                )

                if self.save_txt:
                    self.save_txt_file(path, predictions[0].boxes)
                
                pbar.update()  

        if self.annotate:
            print("Opening labelImg...")
            import os
            os.system(f"labelImg")
        return

    def semi_predict(self):
        import os 
        
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

        
        if self.save_txt:
            print('Making up classes txt files.')
            self.make_up_classes_txt()

        num_transfer = 0
        num_predict = 0 
        with tqdm(total=len(path_list), position=0, leave=True) as pbar:
            for path in path_list:
                txt_name = os.path.basename(path).replace(os.path.splitext(path)[1], '.txt')
                txt_path = os.path.join(os.path.dirname(os.path.dirname(path)), 'labels', txt_name)
                class_set = set()
                txt_lines = []
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        txt_lines = f.readlines()
                        for line in txt_lines:
                            class_set.add(int(line.split()[0]))
                # print(class_set)
                # print(txt_lines)
                if not any(element in class_set for element in self.map_dict.values()):
                    pbar.set_description_str(f"Predicting {os.path.basename(path)}" )
                    predictions = self.model.predict(
                        path,
                        save=self.save,
                        save_txt=None,
                        conf=self.conf,
                        project=project,
                        name=name,
                        exist_ok=True,
                        classes=list(self.class_dict.keys())
                    )
                    if self.save_txt:
                        self.save_semi_txt_file(txt_path, txt_lines)
                        self.save_txt_file(path, predictions[0].boxes)
                    num_predict += 1
                else:
                    # copy to output folder
                    pbar.set_description_str(f"Transfering {os.path.basename(path)}" )
                    if self.save_txt:
                        self.save_semi_txt_file(txt_path, txt_lines)
                    num_transfer += 1
                pbar.update()  

        print(f"Predicted {num_predict} images and transfered {num_transfer} images.")
                                
        if self.annotate:
            print("Opening labelImg...")
            import os
            os.system(f"labelImg")
        return

    def main(self):
        if self.opt.semi:
            self.semi_predict()
        else:
            self.predict()


# python -m YOLOSH --className Predictor --save_txt --includeImg --annotate --weightsPred C:\Users\USER\Projects\20231019-traffic-management\src\assets\weights\yolov8\weights\yolov8l.pt --dataPred C:\Users\USER\Projects\20231019-traffic-management\src\YOLOSH\yaml\train\predict.yaml --map_cls
# python -m YOLOSH --className Predictor --save_txt --includeImg --annotate --weightsPred C:\Users\USER\Projects\20231019-traffic-management\src\YOLOSH\runs\yoloplate-v1l\weights\best.pt --dataPred C:\Users\USER\Projects\20231019-traffic-management\src\YOLOSH\yaml\train\predict.yaml --map_cls
        
        