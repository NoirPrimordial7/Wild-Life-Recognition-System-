import os

def list_folders(path):
    try:
        # Get the list of all entries in the directory
        entries = os.listdir(path)
        
        # Filter out the directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
        
        # Display the names of the folders
        print("Folders in '{}':".format(path))
        for folder in folders:
            print(folder)
    
    except FileNotFoundError:
        print("The specified path does not exist.")
    except PermissionError:
        print("You do not have permission to access this path.")
    except Exception as e:
        print("An error occurred: {}".format(e))

# Specify the path you want to list folders from
path_to_check = r'C:\Users\ADITYA\.cache\kagglehub\datasets\antoreepjana\animals-detection-images-dataset\versions\7\train'
list_folders(path_to_check)