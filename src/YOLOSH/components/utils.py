import yaml

def yaml_to_dict(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

