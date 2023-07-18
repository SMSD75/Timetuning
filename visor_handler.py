import os
import shutil
import zipfile



def extract_zips_in_directories(containing_dir):
    ## recursively extract all zip files in a directory

    for root, dirs, files in os.walk(containing_dir):
        for file in files:
            if file.endswith('.zip'):
                unzip_dir = extract_zip(os.path.join(root, file))
                # change_names(unzip_dir)
                os.remove(os.path.join(root, file))
                print('removed', os.path.join(root, file))
            else:
                print('skipped', os.path.join(root, file))
        
        if dirs == []:
            print('there is no dir', root)
            return
        else:
            print('there is a dir', root)
            for dir in dirs:
                print('dir', dir)
                unzip_dir = os.path.join(root, dir)
                extract_zips_in_directories(unzip_dir)

def change_names(unzip_dir):
    sorted_files = sorted(os.listdir(unzip_dir))
    for i, file in enumerate(sorted_files):
        os.rename(os.path.join(unzip_dir, file), os.path.join(unzip_dir, 'img_{:05d}.jpg'.format(i+1)))


            

def extract_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        dir_name = os.path.dirname(zip_file)
        ## file name without extension
        file_name = os.path.splitext(os.path.basename(zip_file))[0]
        new_dir = os.path.join(dir_name, file_name)
        zip_ref.extractall(new_dir)
        return new_dir


def move_subdirs_to_parent_dir(containing_dir):
    for root, dirs, files in os.walk(containing_dir):
        for dir in dirs:
            for sub_root, sub_dirs, sub_files in os.walk(os.path.join(root, dir)):
                for sub_dir in sub_dirs:
                    shutil.move(os.path.join(sub_root, sub_dir), root)
                shutil.rmtree(sub_root)


def unzip_all_zips_in_dir(containing_dir):
    ## unizp all the zip files in the current directory don't make new directories
    for file in os.listdir(containing_dir):
        if file.endswith('.zip'):
            extract_zip(file)
            os.remove(os.path.join(containing_dir, file))
            print('removed', os.path.join(containing_dir, file))
        elif file.endswith('.jpg'):
            print('skipped', os.path.join(containing_dir, file))
        else:
            os.remove(os.path.join(containing_dir, file))
            print('removed irrelevant', os.path.join(containing_dir, file))

def create_image_dateset(dir):
    sub_dir_list = os.listdir(dir)
    for sub_dir in sub_dir_list:
        sub_dir_path = os.path.join(dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            print(sub_dir_path)
            unzip_all_zips_in_dir(sub_dir_path)

def put_all_zip_files_in_all_subdirs_in_base(based_dir):
    sub_dir_list = os.listdir(based_dir)
    for sub_dir in sub_dir_list:
        sub_dir_path = os.path.join(based_dir, sub_dir)
        ## move all the zip files in the sub_dir to the based_dir
        if os.path.isdir(sub_dir_path):
            for file in os.listdir(sub_dir_path):
                if file.endswith('.zip'):
                    shutil.move(os.path.join(sub_dir_path, file), based_dir)
            shutil.rmtree(sub_dir_path)
        elif os.path.isfile(sub_dir_path):
            if sub_dir_path.endswith('.zip'):
                print('skipped', sub_dir_path)
            else:
                os.remove(sub_dir_path)
                print('removed irrelevant', sub_dir_path)







if __name__ == '__main__':
    # put_all_zip_files_in_all_subdirs_in_base('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/train')
    # put_all_zip_files_in_all_subdirs_in_base('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/val')
    # put_all_zip_files_in_all_subdirs_in_base('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/test')

    # extract_zips_in_directories('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/train')
    # extract_zips_in_directories('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/val')
    # extract_zips_in_directories('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/test')
    # unzip_all_zips_in_dir('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/train')

    # create_image_dateset('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames')
    # extract_zips_in_directories('/ssdstore/ssalehi/VISOR/dataset/Interpolations-DenseAnnotations/val')
    # move_subdirs_to_parent_dir('/ssdstore/ssalehi/VISOR/dataset/GroundTruth-SparseAnnotations/rgb_frames/val')
