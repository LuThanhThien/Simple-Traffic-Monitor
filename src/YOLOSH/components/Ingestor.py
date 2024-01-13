import os
import shutil
from tqdm import tqdm
import random
import time
from components.utils import yaml_to_dict
from ultralytics.yolo.data.utils import IMG_FORMATS

class Ingestor:
    def __init__(self, opt = None) -> None:
        print("Validating input arguments...")
        self.validate_opt(opt)
        self.opt = opt
        time.sleep(1)
        print("Input arguments passed validation.\n")

        print(f"Validating yaml file {opt.dataIngest}")
        self.ingestor_yaml_arg = ['input', 'output', 'relabel']
        self.input, self.output, self.relabel_dict = self.validate_yaml(opt.dataIngest)
        time.sleep(1)
        print("Yaml file passed validation.\n")

        self.img_ext = list(IMG_FORMATS)
        self.mode = opt.mode
        self.ratio = opt.ratio
        self.method = opt.method
        self.ignoreBlank = opt.ignoreBlank
        
        self.blank:int = 0



    def validate_opt(self, opt):
        # validate the input arguments
        if not os.path.exists(opt.dataIngest):
            raise FileNotFoundError(f"Yaml file {self.opt.dataIngest} does not exist.")
        if opt.ratio <= 0:
            raise ValueError("ratio must be greater than 0.")
        if opt.mode not in ['split', 'merge', 'relabel']:
            raise ValueError("mode must be either 'split' or 'merge'.")
        if opt.method not in ['copy', 'cut']:
            raise ValueError("method must be either 'copy' or 'cut'.")
    

    def validate_yaml(self, yaml_path):
        data = yaml_to_dict(yaml_path)
        
        if len(data.keys()) == 0:
            raise KeyError("Yaml file is empty. Please check the yaml file.")

        # output validation
        if 'output' not in data.keys():
            raise KeyError('No output path specified in yaml file.')
        output = data['output']

        if 'input' not in data.keys():
            raise KeyError('No input path specified in yaml file.')
        input = data['input']


        relabel = {}
        if 'relabel' in data.keys():
            if 'path' not in data['relabel'].keys():
                raise KeyError('No path specified in relabel.')
            if 'old' not in data['relabel'].keys():
                raise KeyError('No old specified in relabel.')
            if 'new' not in data['relabel'].keys():
                raise KeyError('No new specified in relabel.')
            relabel = data['relabel']

        if len(data.keys()) > len(self.ingestor_yaml_arg):
            print(f"""WARNING: Yaml file contains more than {len(self.ingestor_yaml_arg)} keys. 
                  Splitter class keys are {self.ingestor_yaml_arg}. 
                  Some keys not recognized in this class will be automatically excluded.""")
            time.sleep(2)
        
        return input, output, relabel
    

    def transfer(self, input, output, file, label_file):
        if self.ignoreBlank:
            with open(os.path.join(input, 'labels', label_file), 'r') as f:
                if len(f.readlines()) == 0:
                    self.blank += 1
                    return
        if self.mode == 'merge':
            file = os.path.join('images', file)
            label_file = os.path.join('labels', label_file)
            
        if self.method == 'copy':
            shutil.copy(os.path.join(input,  file), os.path.join(output, 'images', os.path.basename(file)))
            shutil.copy(os.path.join(input, label_file), os.path.join(output, 'labels', os.path.basename(label_file)))
        elif self.method == 'cut':
            shutil.move(os.path.join(input, file), os.path.join(output, 'images', os.path.basename(file)))
            shutil.move(os.path.join(input, label_file), os.path.join(output, 'labels', os.path.basename(label_file)))


    def summary(self):
        print("Input path: ", self.input)
        print("Output path: ", self.output)
        print("Method: ", self.method)
        print("Ratio: ", self.ratio)
        print('Ignore blank label files: ', self.ignoreBlank)

    
    def split(self):

        self.summary()

        # Create directories for train, valid, and test data
        os.makedirs(os.path.join(self.output, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'val', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'test', 'labels'), exist_ok=True)

        self.blank = 0
        file_log = tqdm(total=0, position=1, bar_format='{desc}')
        
        # Move images and labels to corresponding directories
        img_dir = self.input
        img_pre = ''
        label_pre = ''
        if os.path.exists(os.path.join(self.input, 'images')):
            img_dir = os.path.join(self.input, 'images')
            img_pre = 'images'
        if os.path.exists(os.path.join(self.input, 'labels')):
            label_pre = 'labels'
            
        for file in tqdm(os.listdir(img_dir), "Transfering files"):
            if file.endswith(tuple(self.img_ext)):
                label_file = file.rsplit('.',1)[0] + '.txt'
            else:
                continue
            file = os.path.join(img_pre, file)
            label_file = os.path.join(label_pre, label_file)

            if file.endswith(tuple(self.img_ext)) and os.path.exists(os.path.join(self.input, label_file)):
                file_log.set_description_str(f'Transfering image: {file}')
                # Randomly assign data to train, valid, or test set
                r = random.random()
                if r < self.ratio:
                    output = os.path.join(self.output, 'train')
                else:
                    r = random.random()
                    if r < 0.5:
                        output = os.path.join(self.output, 'test')
                    else:
                        output = os.path.join(self.output, 'val')

                self.transfer(self.input, output, file, label_file)

        if self.blank > 0:
            print(f'Ignored {self.blank} blank label files.')
        
        file_log.set_description_str(f'Finished transfering files.')
        file_log.close()
        print('\n')


    def merge(self):

        self.summary()
        self.blank = 0
        
        # Move images and labels to corresponding directories
        for folder in os.listdir(self.input):
            if os.path.isdir(os.path.join(self.input, folder)): 
                os.makedirs(os.path.join(self.output, folder, 'images'), exist_ok=True)
                os.makedirs(os.path.join(self.output, folder, 'labels'), exist_ok=True)
                
                file_log = tqdm(total=0, position=1, bar_format='{desc}')
                for file in tqdm(os.listdir(os.path.join(self.input, folder, 'images')), "Transfering files to " + folder):
                    label_file = os.path.splitext(file)[0] + '.txt'
                    if file.endswith(tuple(self.img_ext)) and os.path.exists(os.path.join(self.input, folder, 'labels', label_file)):
                        file_log.set_description_str(f'Transfering image: {file}')
                        input = os.path.join(self.input, folder)
                        output = os.path.join(self.output, folder)
                        self.transfer(input, output, file, label_file)
                print('\n')

        if self.blank > 0:
            print(f'Ignored {self.blank} blank label files.')


    def mapping(self):
        mapping = {}
        new_dict = {}
        for key, value in self.relabel_dict['new'].items():
            new_dict[str(value)] = key
        for key, value in self.relabel_dict['old'].items():
            mapping[str(key)] = str(new_dict[value])
        return mapping
    
    def relabel_process(self, text_dir, mapping):
        with tqdm(total=len(os.listdir(text_dir)), position=0, leave=True) as pbar:
            for i, file in enumerate(os.listdir(text_dir)):
                pbar.set_description_str(f'Relabeling file: {file}')
                if file.endswith('.txt'):
                    new_lines = []
                    with open(os.path.join(text_dir, file), 'r') as f:
                        for line in f:
                            new_lines.append(mapping[line.split()[0]] + ' ' + ' '.join(line.split()[1:]) + '\n')
                    with open(os.path.join(text_dir, file), 'w') as f:
                        f.writelines(new_lines)
                        pbar.update()  

    def relabel(self):
        mapping = self.mapping()
        for label_folder in self.relabel_dict['path']:
            text_dir = os.path.join(self.relabel_dict['input'], label_folder)
            print('Relabeling files in ', text_dir)
            self.relabel_process(text_dir, mapping)


    def main(self):
        if self.mode == 'split':
            self.split()
        elif self.mode == 'merge':
            self.merge()
        elif self.mode == 'relabel':
            self.relabel()

