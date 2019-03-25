# convert the dataset to the format of hg-tensorflow
# including miss joint -> -1; use integer joint location; add a index for each mouse in the image
# reference format:
# https://github.com/wbenbihi/hourglasstensorlfow

# TODO: Do we really need boundbox?

import os
from shutil import copyfile

dataset_path = r'/home/eason/Mouse_behavior/2D_Model/trainset_all/trainset.txt'
dataset_tem_path = r'/home/eason/Mouse_behavior/2D_Model/trainset_all/trainset_temp.txt'
copyfile(dataset_path, dataset_tem_path)

try:
    open(dataset_tem_path, 'r')
except IOError as e:
    print("Error! Fail to set back the dataset file")
    raise
    
assert(os.path.isfile(dataset_tem_path)),'Please check path of the dataset file'

with open(dataset_tem_path, 'r') as fs, open(dataset_path, 'w') as fn:
    lines = fs.readlines()
    lines_write = []
    for line in lines:
        line_data = line.strip().split(',')

        # convert the unseen joint from none to -1
        for loc in range(len(line_data)):
            if line_data[loc]=='':
                 line_data[loc]='-1'

        # add[LETTER] for the object considered
        line_data[0] = line_data[0] + r'A'

        # remove the unnecessary number in the decimal
        for loc in range(1,len(line_data)):
            try:
                temp = float(line_data[loc])
            except ValueError as e:
                print("Have occured unvaliable data (not a number)")
                pass
            else:
                line_data[loc] = str(round(temp))

        # convert line_data into line and save with space for split
        line = ' '.join(line_data)
        line = line + '\n' # CR+NL for every line
        lines_write.append(line)
    
    # write into the new file
    fn.writelines(lines_write)

os.remove(dataset_tem_path)

    