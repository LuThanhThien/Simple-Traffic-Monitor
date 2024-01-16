import numpy as np
from typing import List, Union
import dill, copy, random
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator
from src.frame.Detections import Detections
from src.util.utils import *


class FrameTree(Detections):
    def __init__(self, conf_threshold:float=0.5, classes:List[int]=[], class_names:List[str]=[],
                  root:np.ndarray=np.array([]), name:str=None, id:int=-1, parent:'FrameTree'=None):
        super().__init__(conf_threshold, classes, class_names)
        self.root:np.ndarray = root
        self.name:str = name
        self.id:int = id # id from the parent if track, -1 is not track
        self.parent:'FrameTree' = parent
        self.nodes:List['FrameTree'] = []
        

    # get root's components and properxties
    def getRoot(self) -> np.ndarray:
        return self.root
    
    def getName(self) -> str: 
        return self.name
    
    def getBox(self) -> List[int]:
        if self.id == -1:
            return [0,0,self.root.shape[1],self.root.shape[0]] 
        return self.parent.getChildBox(self.id)
    
    def getId(self) -> int:
        return self.id
    
    def getNodes(self) -> List['FrameTree']:
        return self.nodes
    
    def getNodeCount(self) -> int:
        count = len(self.nodes)
        for node in self.nodes:
            count += node.getNodeCount()
        return count

    def getChildCount(self) -> int:
        return len(self.nodes)

    def getDepth(self) -> int:
        if len(self.nodes) == 0:
            return 0
        return max(node.getDepth() for node in self.nodes) + 1

    def getShape(self) -> List[int]:
        return self.root.shape
    
    def getAllParents(self): # all levels
        """
        get all the parents of the current node
        """
        if self.id == -1:
            return []
        parents = []
        parents.append(self.parent)
        parents.extend(self.parent.getAllParents())
        return parents

    def getParent(self, level:int=1) -> 'FrameTree': # access by level
        """
        get the parent of the current node by level:
            - level=1 is the direct parent, level=2 is the grandparent, etc.
            - level=-1 is the root parent, level=-2 is the child of the root parent, etc.
            - level=0 is the current node itself
        """
        if level == 0:
            return self
        parents = self.getAllParents()
        if len(parents) == 0:
            print('This node has no parent')
            return None
        elif level > len(parents):
            raise ValueError(f'Level exceeds upper tree depth {len(parents)}.')
        if level > 0:
           return parents[level-1]
        if level < 0:
            return parents[level]
        
    # get root's child primary level - get 1 child
    def getChild(self, index:int) -> Union['FrameTree', None]:
        if index < len(self.nodes):
            return self.nodes[index]
        print('Index out of range')
        return None
    
    def getChildById(self, id:int) -> Union['FrameTree', None]:
        for node in self.nodes:
            if node.id == id:
                return node
        print('No child with id:', id)
        return None
        
    def getChildByBox(self, box:np.ndarray=np.array([])) -> Union['FrameTree', None]:
        if len(box) == 0:
            raise ValueError('Please specify the box, box has no dimension.')
        for node in self.nodes:
            if np.equal(np.array(node.getBox()),np.array(box)):
                return node
        print('No child with box:', box)
        return None
    
    def getChildBox(self, id:int) -> List[int]:
        for i, child_id in enumerate(self.idx):
            if child_id == id:
                return self.bboxes[i]
        print('No child with id:', id) 
        return None
        
    # get root's child access all levels - get list of chilren
    def getAllChilds(self, max_depth:int=-1) -> List['FrameTree']:
        if max_depth > self.getDepth():
            raise ValueError(f'Maximum depth exceeds tree depth {self.getDepth()}.')
        if max_depth < -1:
            raise ValueError('Maximum depth must be either non-negative or -1 (get all children in tree).')
        if max_depth == -1:
            max_depth = self.getDepth()
        elif max_depth == 0:
            return []
        childs = []
        for node in self.nodes:
            childs.append(node)
            if node.getDepth() > 0:
                childs.extend(node.getAllChilds(max_depth=max_depth-1))
        return childs

    def getChildsByLevel(self, level:int=1) -> List['FrameTree']:
        childs = []
        if level == 1:
            return self.getAllChilds(max_depth=1)     
        if level > self.getDepth():
            raise ValueError(f'Level exceeds tree depth {self.getDepth()}.')
        elif level <= 0:
            raise ValueError('Level must start from 1.')
        for node in self.nodes:
            if node.getDepth() > 0:
                childs.extend(node.getChildsByLevel(level=level-1))
        return childs
    
    def getChildsByName(self, name:str) -> List['FrameTree']:
        childs = []
        for node in self.nodes:
            if node.name == name: childs.append(node)
            if node.getDepth() > 0:
                childs.extend(node.getChildsByName(name=name))
        return childs
    

    # save and load
    def save(self, path:str=None) -> None:
        if path == None:
            path = './cascades'
            _, next_file = latest_version_file(path, '.pkl')
            path = os.path.join(path, next_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump(self, f)
        print('Frame Tree saved to:', '\033[1m' + path + '\033[0m')

    @classmethod
    def load(cls, path:str) -> 'FrameTree':
        print('Loading Frame Tree from:', '\033[1m' + path + '\033[0m')
        with open(path, 'rb') as f:
            loaded_tree = dill.load(f)
        instance = cls()
        instance.__dict__.update(loaded_tree.__dict__)
        return instance

    def copy(self) -> 'FrameTree':
        return copy.copy(self)

    def deepcopy(self) -> 'FrameTree':
        return copy.deepcopy(self)       

    # superclass callbacks
    def super_save(self, path:str='detections\detect0.pkl') -> None:
        return super().save(path)
    
    def super_load(self, path:str) -> 'Detections':
        return super().load(path)

    # data-adjustment methods
    def append_node(self, node:'FrameTree') -> None:
        self.nodes.append(node)
        
    # add branches
    def add(self, results:Results, get_max:bool=False) -> None:
        """
        add nodes to a root node, results is Results object from Ultrlytics engine
        """
        for r in results:
            for b in r.boxes:
                # append to Detections object
                detection = super().append_box(b)
                if detection == None:
                    continue
                # crop
                bbox, conf, id, name = detection
                x1, y1, x2, y2 = bbox
                crop_frame = np.ascontiguousarray(self.root[y1:y2, x1:x2])
                self.append_node(FrameTree(root=crop_frame, name=name, id=id, parent=self))

                # https://github.com/ultralytics/ultralytics/issues/4971
                if get_max:
                    break

    # tree develop and union
    def is_same_root(self, compared_frame_nodes:'FrameTree') -> bool:
        """
        check whether two trees have the common node 
        """
        frame = compared_frame_nodes.getRoot()
        condition = np.array_equal(self.root, frame)
        if self.parent: 
            condition |= self.parent.is_same_root(compared_frame_nodes)
        if compared_frame_nodes.parent:
            condition |= compared_frame_nodes.parent.is_same_root(self)
        return condition

    @staticmethod    
    def iob_matched_idx(candidates:np.ndarray, objects:np.ndarray) -> np.ndarray:
        """
        calculate the IOB score for 2 matrix of bounding boxes (xyxy) in matrix-multiplication-like way
        """
        scores = [[IOB(cand, obj) for obj in objects] for cand in candidates]
        return np.array(scores)

    @staticmethod
    def argmax_matrix2D(array:np.ndarray, axis:int, conf_threshold:float) -> np.ndarray:
        """
        return a boolean matrix with the maximum value in each row or column for a 2D array,
        and whether that maximum value is greater than or equal to a given conf_threshold
        """
        array = np.atleast_2d(array)
        # argmax_array = np.argmax(array, axis=axis)
        max_values = np.max(array, axis=axis, keepdims=True)
        result = (array == max_values) & (max_values >= conf_threshold)
        return result

    @staticmethod
    def narrowBox(parent_box:list, child_box:list) -> List[int]:
        """
        re-coordinate the child box wrt the parent box coordinate
        """
        x1r, y1r, x2r, y2r = [child_box[i] - parent_box[i%2] for i in range(4)]
        return [x1r, y1r, x2r, y2r]
    
    def flattenBox(self) -> List[int]:
        """
        re-coordinate the child box wrt the parent box coordinate
        """
        if self.id == -1:
            return [0,0,self.root.shape[1],self.root.shape[0]]
        return [self.getBox()[i] + self.parent.getBox()[i%2] for i in range(4)]

    def develop(self, child_node:'FrameTree', child_indentity:list) -> None:
        """
        branch a tree with a child node and its detections from different tree with the common node
        """
        if not self.is_same_root(child_node): 
            raise Exception('Cannot concat trees with different roots')
        self.nodes.append(child_node)
        self.append(*child_indentity)
        
        this_bbox = [0,0,self.root.shape[1],self.root.shape[0]] if (self.id==-1)\
              else self.parent.getChildBox(self.id) 
        child_bbox = child_node.parent.getChildBox(child_node.id)
        self.bboxes[-1] = self.narrowBox(this_bbox, child_bbox) # re-coordinate
        self.nodes[-1].parent = self # change parent

    def union(self, node2:'FrameTree') -> None:
        """
        within the common node, union two trees
        """
        if not self.is_same_root(node2):
            raise Exception('Cannot concat trees with different roots')
        self.nodes.extend(node2.nodes)
        self.concat(node2)

    def deepunion(self, frame_nodes:'FrameTree', conf_threshold=0.5, deepcopy=True) -> 'FrameTree':
        """
        within the common node, union two trees and detect their corresponding relationship
        """
        if not self.is_same_root(frame_nodes):
            raise Exception('Cannot concat trees with different roots')
        
        # get the larger tree and the smaller tree
        Node1Num, Node2Num = self.getChildCount(), frame_nodes.getChildCount()
        lNode:'FrameTree' = self.deepcopy() if Node1Num>=Node2Num else frame_nodes.deepcopy()
        sNode:'FrameTree' = frame_nodes.deepcopy() if Node1Num>=Node2Num else self.deepcopy()
        # get the IOB matrix and the argmax matrix
        iob_matrix = self.iob_matched_idx(sNode.bboxes, lNode.bboxes)
        # print('The iob matrix is', iob_matrix)
        # if no detections
        if len(iob_matrix) == 0:
            return lNode

        iob_argmatrix = self.argmax_matrix2D(iob_matrix, 1, conf_threshold) & self.argmax_matrix2D(iob_matrix, 0, conf_threshold)
        # print('This is iob argmatrix', iob_argmatrix)
        # develop the larger tree with the smaller tree
        for i, row in enumerate(iob_argmatrix): 
            lId = np.where(row)[0]
            sId = i
            if len(lId) != 0:
                # print('union to child')
                lId = lId[0]
                lChildNode:'FrameTree' = lNode.nodes[lId]
                lChildNode.develop(sNode.nodes[sId], sNode.getIdentity(sId))
                lNode.nodes[lId] = lChildNode
            else:
                # print('union to root')
                lNode.develop(sNode.getChild(sId), sNode.getIdentity(sId))
        if not deepcopy: 
            print('Warning: this tree is unioned without deep copy')
            self = lNode
        return lNode


    # summarize and plot
    def draw_tree(self, depth:int=0, id:int=-1, is_last_sibling:bool=False) -> str:
        """
        draw the tree in a string format
        """
        prefix = ''
        prefix = '    ' * (depth - 1) if depth > 0 else prefix
        prefix += ('└── ' if is_last_sibling else '├── ') if depth > 0 else ''
        prefix = prefix + f'[{id}] ' if id != -1 else prefix
        tree = prefix + str(self.name) + f'\n'
        for i, child in enumerate(self.nodes):
            id = child.id
            child_prefix = '│   ' * depth if not is_last_sibling else '    ' * depth
            tree += child_prefix + child.draw_tree(depth + 1, id, i == len(self.nodes) - 1)
        return tree
    
    def summarize(self) -> None:
        """
        summarize the tree
        """
        from colorama import Fore
        print()
        print("\033[1m" + Fore.YELLOW + 'FrameTree Summary' + "\033[0m")
        print('Root Name: ', self.name)
        print('Root ID: ', self.id)
        print('Root Shape: ', self.getShape())
        print('Root Child Count: ', self.getChildCount())
        print('Root Depth: ', self.getDepth())
        print('Tree Diagram: ')
        print(self.draw_tree())


    def plot(self, annotator:Annotator=None, root_frame:np.ndarray=np.array([]), root_coordinate:List[int]=[0,0], color=(230, 23, 59), seperate=False, verbose=False, **krargs) -> np.ndarray:
        """
        plot the tree wrt the root coordinate
        """
        # copy the root frame
        if len(root_frame) == 0:  
            annotate_frame = self.root.copy()
            annotator = Annotator(annotate_frame, **krargs)
        else:
            annotate_frame = root_frame.copy()
            annotator = annotator

        # loop through the children
        for i in range(self.getChildCount()):
            # get the child and its detection
            child = self.getChild(i)
            bbox, conf, id, name = self.getIdentity(i)
            if (bbox == None): continue

            # transform the bbox to the root coordinate and plot
            bbox_trans = [bbox[j]+root_coordinate[j%2] for j in range(4)]

            if seperate and self.id == -1:
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            annotator.box_label(bbox_trans, f'ID: {id} {name} {int(conf*100)}%', color=color)

            if verbose:
                print(f'ID: {id:<3} {name:<15} {int(conf*100):<3}% bbox (xyxy): {bbox}')

            # recursive plot
            child_coordinate = [bbox_trans[0], bbox_trans[1]]
            annotate_frame = child.plot(annotator, annotate_frame, child_coordinate, color=color, verbose=verbose, seperate=seperate)

        annotate_frame = annotator.result()
        return annotate_frame
