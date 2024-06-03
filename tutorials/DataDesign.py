"""
03.06.2024
Lucas Steinberger, DKFZ
python file with pseudo-code for loading data from the DKFZ tissue atlas dataset using the existing htc framework
The htc framework was designed and implemented with a much simpler data/folder structure than the tissue atlas,
as a result we are working to write code that can make the htc framework compatible with the tissue atlas structure
furthermore, we hope to create a .py or .ipynb file that takes users less experienced in coding through easy 
steps to use the htc framework to train and use a segmentation model on their own desired selection from the tissue atlas dataset

!!!this code is currently only to assist the developer in design purposes!!!
"""

#define a path to a dataset_settings.json file. in the original framework, this .json lives in the "data" folder,
#but because there may be more than one folder the user wishes to include as data, here we define it Seperately. 
jsonpath = "path/to/my/dataset_settings.json"
#NOTE: importantly, the json should still be named "dataset_settings.json". currently,
#another name such as "my_project.json" will not work (plan to hopefully change this)


#create a list of all the desired data directories that you want the htc to load

###should this be done using path environment?
directories_list = ["path/to/directory/1", "path/to/directory/2",] #fill with as many directories as desired
paths = [] #initialize paths
for directory in directories_list:
    addpaths = list(DataPathAtlas.iterate())
    paths .append(addpaths)
    """
    The desired end result here is a variable "paths" that contains a list of Path objects, with one path object to every image folder
    that you want to be included in the dataset. the Iterate method of DataPathAtlas will be custom designed, so that it will keep going down the folder
    "tree" until it finds the image folders. this is to deal with the fact that not all image folders are at the same depth in the overall tissue atlas dataset.
    it will still be possible to further narrow down the data you want by using the filters present in the iterate method.
    """
    
