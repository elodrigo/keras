import datetime
import uuid
import os


def unique_filename(type='uuid'):
    if type == 'datetime':
        filename = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    else:  # type == "uuid"
        filename = str(uuid.uuid4())
    return filename


def makenewfold(prefix='output_', type='datetime'):
    suffix = unique_filename('datetime')
    foldname = 'model/' + suffix
    os.makedirs(foldname)
    return foldname


def what_is_newest_folder():
    list_of_folders = next(os.walk('./model'))[1]
    list_of_folders.sort(reverse=True)
    folder_name = 'model/' + list_of_folders[0]
    return folder_name
