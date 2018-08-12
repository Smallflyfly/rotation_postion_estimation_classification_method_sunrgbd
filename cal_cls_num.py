import numpy as np
import xml.etree.ElementTree as ET
import os
  
  
def load_pascal_annotation(filename, total_num):

    all_cls = ('__background__',  # always index 0
                    'chair', 'table', 'sofa', 'bed', 'shelf', 'cabinet')
    # w , h = self.get_pic_size(picname)
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)
    if num_objs > 1:
        return 0
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        clsname = obj.find('name').text.lower().strip()
        if clsname == 'cabinet':
            # clsname = 'table'
            return 1
        else:
            return 0
        # ind = all_cls.index(clsname)
        # total_num[ind] += 1

xmlpath = './data/VOCdevkit2007/VOC2007/Annotations/'
xmlfiles = os.listdir(xmlpath)
total_num = [0, 0, 0, 0, 0, 0, 0]
total_num = 0
for xmlfile in xmlfiles:
    num = load_pascal_annotation(xmlpath + xmlfile, total_num)
    if num > 0:
        total_num += 1
        print(xmlfile)
print(total_num)
print('Done')
# print(total_num)