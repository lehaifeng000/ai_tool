# tt100k 2021转yolo训练格式

import os
from pathlib import Path
import json
import shutil
import cv2

class TT2YOLO:
    def __init__(self,tt100k_path=".",target_path=".") -> None:
        '''
        tt100k_path : tt100k解压路径
        target_path : 目标数据集存放路径
        '''
        self.raw_path=Path(tt100k_path)
        self.target_path=Path(target_path)
        
        pass
    
    
    def filter_class(self,):
        '''
        过滤图像不足100张的目标类别
        '''
        pass
        
        # 读TT100K原始数据集标注文件
        anno_path = self.raw_path.joinpath('annotations_all.json')
        origin_dict = json.loads(anno_path.read_text())
        classes = origin_dict['types']
        # with open(os.path.join(self.tt100k_path, 'annotations.json')) as origin_json:
        #     origin_dict = json.load(origin_json)
        #     classes = origin_dict['types']
        
        # 建立统计每个类别包含的图片的字典
        sta = {}
        for i in classes:
            sta[i] = []
        images_dic = origin_dict['imgs']
        
        # 记录所有保留的图片
        saved_images = []
        # 遍历TT100K的imgs
        for image_id in images_dic:
            image_element = images_dic[image_id]
            image_path = image_element['path']
            if not (image_path.startswith('train') or image_path.startswith('test') ):
                continue

            # 添加图像的信息到dataset中
            # image_path = image_path.split('/')[-1]
            obj_list = image_element['objects']

            # 遍历每张图片的标注信息
            for anno_dic in obj_list:
                label_key = anno_dic['category']
                # 防止一个图片多次加入一个标签类别
                if image_path not in sta[label_key]:
                    sta[label_key].append(image_path)

        # 只保留包含图片数超过100的类别（重新划分，阈值100可根据需求修改）
        result = {k: v for k, v in sta.items() if len(v) >= 100}

        for i in result:
            print("the type of {} includes {} images".format(i, len(result[i])))
            saved_images.extend(result[i])

        saved_images = list(set(saved_images))
        print("total types is {}".format(len(result)))

        type_list = list(result.keys())
        result = {"type": type_list, "details": result, "images": saved_images}
        print(type_list)
        self.type_list=type_list
        self.images=saved_images
        self.details=result
        pass
        # # 保存结果
        # json_name = os.path.join(parent_path, 'statistics.json')
        # with open(json_name, 'w', encoding="utf-8") as f:
        #     json.dump(result, f, ensure_ascii=False, indent=1)

    def gen_labels(self,):
        '''
        生成标签txt
        '''
        def convert(size, box):
            # dw = 1. / (size[0])
            # dh = 1. / (size[1])
            igw=size[0]
            igh=size[1]
            
            x = 0.5*(box[0] + box[2]) /igw
            y = 0.5*(box[1] + box[3]) / igh
            w = min(box[2]/igw,1.0)
            h = min(box[3]/igh,1.0)
            # round函数确定(xmin, ymin, xmax, ymax)的小数位数
            x = round(x , 6)
            if x>1:
                x=1.0
            w = round(w , 6)
            y = round(y , 6)
            if y>1:
                y=1.0
            h = round(h , 6)
            return (x, y, w, h)
        
        
        # 读TT100K原始数据集标注文件
        anno_path = self.raw_path.joinpath('annotations_all.json')
        origin_dict = json.loads(anno_path.read_text())
        labels_path=self.target_path.joinpath('labels')
        labels_path.mkdir(parents=True,exist_ok=True)
        
        for img_path in self.images:
            file_name=img_path.split('/')[-1]
            img_id=file_name.split('.')[0]
            objs=origin_dict['imgs'][img_id]['objects']
            f_txt=labels_path.joinpath(img_id+'.txt')
            f_txt.touch(exist_ok=True)
            lines=[]
            if img_id=='15527':
                pass
                aaa=1
            for obj in objs:
                if obj['category'] not in self.type_list:
                    continue
                label_id=self.type_list.index(obj['category'])
                bbox=obj['bbox']
                box=[bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax'],]
                x,y,w,h=convert((2048,2048),box)
                # f_txt.write_text("%s %s %s %s %s\n" % (label_id, x, y, w, h))
                lines.append("%s %s %s %s %s" % (label_id, x, y, w, h))
                pass
            f_txt.write_text(self.conbine_lines(lines))
            pass
        pass
        
        
        
    def cp_img(self,):
        '''
        复制图像
        '''
        for img_path in self.images:
            total_row_img_path = str(self.raw_path.joinpath(img_path))
            file_name=img_path.split('/')[-1]
                
            total_target_path=self.target_path.joinpath('images',file_name)
            total_target_path.parent.mkdir(parents=True, exist_ok=True)
            # 复制图像
            # shutil.copyfile(total_row_img_path,str(total_target_path))
            # 读取图像，resize为640大小，训练会快很多
            img0=cv2.imread(total_row_img_path)
            img0=cv2.resize(img0,(640,640))
            cv2.imwrite(str(total_target_path),img0)
            pass
            
        
        
    def split(self,):
        '''
        划分训练验证，生成txt
        '''
        train_lines=[]
        test_lines=[]
        for img_path in self.images:
            file_name=img_path.split('/')[-1]
            if img_path.startswith('train'):
                train_lines.append('./images/'+file_name)
            else:
                test_lines.append('./images/'+file_name)
        train_split=self.target_path.joinpath('train.txt')
        train_split.touch(exist_ok=True)
        train_split.write_text(self.conbine_lines(train_lines))
        
        test_split=self.target_path.joinpath('test.txt')
        test_split.touch(exist_ok=True)
        test_split.write_text(self.conbine_lines(test_lines))
    
    def yaml(self,):
        y8yaml=self.target_path.joinpath('tt100k.yaml')
        y8yaml.touch(exist_ok=True)
        lines=[]
        lines.append('path: ./datasets/tt100k')
        lines.append('train: train.txt')
        lines.append('val: test.txt')
        lines.append('test: test.txt')
        lines.append('names:')
        for i, name in enumerate(self.type_list):
            lines.append(" "+str(i)+": "+name)
        y8yaml.write_text(self.conbine_lines(lines))
        
    
    def conbine_lines(self,lines):
            l=len(lines)
            s=''
            for i,line in enumerate(lines):
                s=s+line
                if i!=l-1:
                    s=s+'\n'
            return s 
    
if __name__=='__main__':
    print("start")
    # 这两个路径需要替换
    ttyolo=TT2YOLO(tt100k_path='/home/lhf/tt100k_2021',target_path='/home/lhf/src/python/ai_tool/dataset/tt100k/tt100k_yolo')
    ttyolo.filter_class()
    ttyolo.cp_img()
    ttyolo.gen_labels()
    ttyolo.split()
    ttyolo.yaml()
    pass
        