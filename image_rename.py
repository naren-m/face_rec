#############################
# Author: Naren Mudivarthy
#
#############################

# Script to rename image files with timestamp of image creation

import os, time, datetime
import subprocess, re, sys
import shutil
import argparse

from PIL import Image

import logging
# formatter = '[%(asctime)s] {%(pathname)s} %(funcName)s:%(lineno)d %(levelname)s - %(message)s','%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG,
                    format= "[%(asctime)s] {%(pathname)s} %(funcName)s:%(lineno)d %(levelname)s - %(message)s",
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                    )

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

image_extensions = list()

image_extensions.append(".jpg")
image_extensions.append(".png")
image_extensions.append(".gif")
image_extensions.append(".JPG")
image_extensions.append(".PNG")
image_extensions.append(".GIF")
image_extensions.append(".CR2")
image_extensions.append(".cr2")

def get_jpeg_exif_time(file_path):
    if not os.path.isfile(file_path):
        return None
    try:
        image = Image.open(file_path)
        if hasattr(image, '_getexif'):
            exifdata = image._getexif()
            create_time = exifdata[0x9003]
            return create_time
    except:
        _type, value, traceback = sys.exc_info()
        logging.error("Error:\n{}".format(value))

    return None

"""
Generates new file name for an image with the timestamp of image creation time
"""
def generate_filename_with_timestamp(file_path):
    new_file_name = get_jpeg_exif_time(file_path)
    if not new_file_name:
        logging.error("Couldn't find file creation time for - {}".format(
                        file_path))
        return new_file_name

    new_file_name = new_file_name.replace("-", "_")
    new_file_name = new_file_name.replace(":", "_")
    new_file_name = re.sub('[^\d]+', '_', new_file_name)
    return new_file_name

"""
Renaming images to creation time
"""
def rename_images(image_folder_path, destination_dir = None,
                    move_to_dest = False):
    file_name_dict = dict()

    # for mac, cleaning up the .DS_Store files
    logging.info("started cleaning .DS_Store files")
    find_and_delete_ds_store = "find . -type f -name '.DS_Store'  -delete"
    subprocess.call(find_and_delete_ds_store, shell=True)
    logging.info("completed cleaning .DS_Store files")

    # walk through the directories
    total_files_processed = 0
    skipped_file_count = 0
    for walk_dir, dirs, files in os.walk(image_folder_path):
        for file in files :
            f, file_extension = os.path.splitext(file)
            # skip if file in not an image
            if not file_extension in image_extensions:
                logging.info("File {} is not an image. Skipping process".format(f))
                continue

            file_path = os.path.join(walk_dir, file)
            new_file_name = generate_filename_with_timestamp(file_path)

            if not new_file_name:
                logging.error("Unable to find creation time for {}{}".format(
                        f, file_extension ))
                skipped_file_count+=1
                continue

            # Handling the case for image with same time stamp
            if new_file_name in file_name_dict.keys():
                file_name_dict[new_file_name] +=1
                new_file_name = new_file_name + "_" + \
                                    str(file_name_dict[new_file_name])
            file_name_dict[new_file_name] = 0

            if destination_dir is None:
                dest_dir = walk_dir
            else :
                dest_dir = destination_dir
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

            if file_extension == ".CR2":
                logging.info("Found CR2 image - {}{}".format(f, file_extension))
                if "CR2" in walk_dir:
                    logging.info("Skipping creating another CR2 in {}".format(
                                    dest_dir))
                    continue

                dest_dir = os.path.join(walk_dir, file_extension[1:])

                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

            new_file_path = os.path.join(dest_dir, new_file_name + file_extension.lower())
            logging.info("Renaming {0} to {1}{2}".format(file_path,
                    new_file_name,file_extension))

            if move_to_dest:
                try:
                    shutil.move(file_path, new_file_path)
                except Exception as e:
                    logging.error("Couldn't perform " +file_path + "->" +new_file_name)
                    logging.error(e)
                    skipped_file_count+=1
            else:
                try:
                    shutil.copy(file_path, new_file_path)
                except Exception as e:
                    logging.error("Couldn't perform " +file_path + "->" +new_file_name)
                    skipped_file_count+=1
                    logging.error(e)

            total_files_processed+=1
    return total_files_processed, skipped_file_count

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbose', default=False,
                        help='logging.infos logs in detail')

    parser.add_argument('-i', '--input', required=True,
                        help='Root directory for images to process')

    parser.add_argument('-o', '--output',
                        help='Destination directory for all the images (if not exists creates)')

    parser.add_argument('-m', '--move', default=False,
                        help='Moves images from the source dir to destination. Default is false')

    args = parser.parse_args()
    path = args.input

    try:
        if(not os.path.isdir(path)):
            logging.error("Invalid directory {} ".format(path))
            logging.error("Exiting")
            sys.exit(1)
    except Exception as e:
        logging.error("Caught exception {} ".format(str(e)))

    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logging.info("Input directory{}".format(args.input))
    logging.info("Output directory{}".format(args.output))
    logging.info("Move to {} - {}".format(args.output, args.move))

    # Processing files
    processed_files, skipped_files = rename_images(path, args.output, args.move)

    logging.info("Processed {} images. Skipped {} images".format(
                processed_files, skipped_files, ))

"""
    Input Parameters
        verbose : Prints logs in detail
        input   : Root directory for images to process
        output  : Destination directory for all the images (if not exists creates)
        move    : Moves images from the source dir to destination. Default is false
"""
if __name__ == '__main__':
    main()
