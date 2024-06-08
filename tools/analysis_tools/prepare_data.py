import os
import shutil

def copy_tif_files(source_dir, destination_dir):
    try:
        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        for img_dir in os.listdir(source_dir):
            
            # Get a list of files in the source directory
            files = os.listdir(source_dir + img_dir)
            
            # Iterate through each file and copy if it has a .tif extension
            for file in files:
                if file.endswith('.tif'):
                    
                    source_file_path = os.path.join(source_dir + img_dir, file)
                    # some files have the same name, use directory name with filename as output file
                    destination_file_path = os.path.join("D:/" + destination_dir, img_dir + '_' + file )
                    print(destination_file_path)
                    shutil.copy(source_file_path, destination_file_path)
                    print(f"File '{file}' copied successfully.")
            
            print("All .tif files copied successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:

source_directory = 'unzipped_3/'
destination_directory = '/images_batch_3'

copy_tif_files(source_directory, destination_directory)




