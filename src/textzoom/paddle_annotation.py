import os

TEXTZOOM_PATH = r'C:\Users\USER\Projects\20231019-traffic-management\src\assets\data\TextZoomData'
train_path = os.path.join(TEXTZOOM_PATH, 'train')
val_path = os.path.join(TEXTZOOM_PATH, 'val')
train_txt_path = os.path.join(TEXTZOOM_PATH, 'train.txt')
val_txt_path = os.path.join(TEXTZOOM_PATH, 'val.txt')

mapping_path = {
   train_path: train_txt_path,
   val_path: val_txt_path
}

def preprocess_dir():
   for path in [train_path, val_path]:
      return      # done already
      for root, dirs, files in os.walk(r"C:\Users\USER\Projects\20231019-traffic-management\src\assets\data\TextZoomData\val\HRx2"):
         for file in files:
            if file.endswith('.png'):
               img_path = os.path.join(root, file)
               os.rename(img_path, os.path.join(os.path.dirname(root), 'HRx2_' + file))
         pass

      for root, dirs, files in os.walk(path): 
         for dir in dirs:
            if '_' in dir:
               subfolder = dir.split('_')[-1]
               out_path = os.path.join(path, subfolder)
               print(out_path)
               os.makedirs(out_path, exist_ok=True)

               for root, dirs, files in os.walk(os.path.join(path, dir)):
                  for file in files:
                     if file.endswith('.png'):
                        img_path = os.path.join(root, file)
                        os.rename(img_path, os.path.join(out_path, file))
               
      
def to_paddle_txt():
   for path, txt_path in mapping_path.items():
      if os.path.exists(txt_path):
         os.remove(txt_path)

      for root, dirs, files in os.walk(path):
         for file in files:
            if 'LR' in os.path.join(root, file):
               continue
   
            if '_' in file:
               img_path = os.path.join(root, file) 
               file = strip_name_space(img_path)
               name = label_extract(file)
               relative_path = path_extract(img_path)
               line = relative_path + ' "' + name + '"\n'
               with open(txt_path, 'a', encoding="utf-8") as f:
                  f.write(line)

def label_extract(file):
   return file.rsplit('.', 1)[0].split('_')[-2]

def path_extract(path:str):
   return path.split(TEXTZOOM_PATH)[-1][1:].replace('\\', '/')

def strip_name_space(file:str):
   name = os.path.basename(file)
   new_name = name.replace(' ', '')
   os.rename(file, os.path.join(os.path.dirname(file), new_name))
   return new_name


if __name__ == '__main__':
   # preprocess_dir()
   to_paddle_txt()