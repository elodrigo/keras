import datetime
import os


def what_is_newest_folder():
    list_of_folders = next(os.walk('./model'))[1]
    list_of_folders.sort(reverse=True)
    folder_name = 'model/' + list_of_folders[0]
    return folder_name

