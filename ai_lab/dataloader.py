import ai_lab.storagemanager as sm
from .datasets import density_map_dot
from .datasets import density_map_dot_experimental
from .datasets import image_folder

# Translates dataset_type into dataset class
dataset_type_translation_table = {
	"DENSITY_MAP_DOT": density_map_dot.DensityMapDot,
	"DENSITY_MAP_DOT_EXPERIMENTAL": density_map_dot_experimental.DensityMapDot,
	"IMAGE_FOLDER": image_folder.ImageFolder
}

'''
	load_dataset(path[dir/file], dataset_type,args[optional dict (e.g. gauss-kernel-radius)])

	Check if dataset_type exists 
	and if path(file or dir) exists

	Then automatically loads the right dataset-class
	and returns it initialized with the path.
	(All dataset-classes must only take a file or dir
	path as initialization parameter)
'''
def load_dataset(path, dataset_type,args=[]):
	print "[1]"
	if not dataset_type in dataset_type_translation_table:
		return None
	path = sm.check_dir_or_file_path(path)
	if path == None:
		return None


	print "[2]"
	dataset = dataset_type_translation_table[dataset_type]

	return dataset(path,args=args)