import os

# None if file doesnt exist
# abspath(file) if it exists
def check_file_path(path):
	if os.path.isfile(os.path.abspath(path)):
		return os.path.abspath(path)
	return None

# None if directory doesnt exist
# abspath(directory) if it exists
def check_dir_path(path):
	if os.path.isdir(os.path.abspath(path)):
		return os.path.abspath(path)
	return None

# None if directory and or file doesnt exist
# abspath(directory and or file) if it exists
def check_dir_or_file_path(path):
	if os.path.isdir(os.path.abspath(path)) or os.path.isfile(os.path.abspath(path)):
		return os.path.abspath(path)
	return None


# True/False file exists
def file_exists(path):
	if os.path.isfile(os.path.abspath(path)):
		return True
	return False

# True/False directory exists
def dir_exists(path):
	if os.path.isdir(os.path.abspath(path)):
		return True
	return False

# True/False directory and or file exists
def dir_or_file_exists(path):
	if os.path.isdir(os.path.abspath(path)) or os.path.isfile(os.path.abspath(path)):
		return True
	return False
