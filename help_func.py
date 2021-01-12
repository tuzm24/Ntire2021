import math
import numpy as np
import os
import csv



class myUtil(object):

    @staticmethod
    def removeExtinFolder(path):
        for filename in os.listdir(path):
            print(os.path.join(path,filename))
            os.rename(os.path.join(path, filename),
                      os.path.join(path, os.path.splitext(filename)[0]))
    @staticmethod
    def removeExtinFile(dir, ext):
        for (path, dirnames, files) in os.walk(dir):
            for filename in files:
                if ext in filename:
                    os.rename(os.path.join(path, filename),
                              os.path.join(path, filename.replace(ext, '')))
    @staticmethod
    def static_vars(**kwargs):
        def decorate(func):
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func
        return decorate

    @staticmethod
    def psnr(loss):
        if loss <= 0:
            return 100
        return math.log10(1 / loss) * 10

    @staticmethod
    def getMAEfromNumpy(A, B):
        return np.abs(np.subtract(A, B)).mean()

    @staticmethod
    def UpSamplingChroma(UVPic):
        return UVPic.repeat(2, axis = 0).repeat(2, axis = 1)

    @staticmethod
    def getMSEfromNumpy(A, B):
        return np.square(np.subtract(A, B)).mean()


    @staticmethod
    def getFileList(dir, ext='.bin'):
        matches = []
        # for root, dirnames, filenames in os.walk(dir):
        #     for filename in filenames:
        for filename in os.listdir(dir):
            if filename.endswith(ext):
                matches.append(os.path.join(dir, filename))
        return matches



    @staticmethod
    def xgetFileList(dir, ext='.bin'):
        matches = []
        for (path, dirnames, files) in os.walk(dir):
            for filename in files:
                if filename.endswith(ext):
                    matches.append(os.path.join(path, filename))
        return matches


    @staticmethod
    def getDirlist(path):
        flist = os.listdir(path)
        returnlist = []
        for fpath in flist:
            if os.path.isdir(os.path.join(path, fpath)):
                returnlist.append(os.path.join(path, fpath))
        return returnlist

    @staticmethod
    def filteringPath(file_list, word):
        for f in file_list:
            if word in os.path.basename(f):
                return f

    @staticmethod
    def xgetFileNum(dir, ext='.bin'):
        matches = []
        for (path, dirnames, files) in os.walk(dir):
            for filename in files:
                if filename.endswith(ext):
                    matches.append(os.path.join(dir, filename))
        return len(matches)

    @staticmethod
    def initHeaderCSV(dir, header):
        with open(dir, 'w', newline='') as f:
            headerlinewriter = csv.writer(f)
            headerlinewriter.writerow(header)

    @staticmethod
    def intoRealFolder(dir):
        dirs = myUtil.getDirlist(dir)
        if len(dirs)==0:
            return os.path.basename(os.path.dirname(dir))
        return myUtil.intoRealFolder(dirs[0])

    @staticmethod
    def getBaseDirnameAndBasename(dir):
        return os.path.join(os.path.basename(os.path.dirname(dir)), os.path.basename(dir))

    @staticmethod
    def getleafDirs(dir):
        folders = []
        for root, dirs, files in os.walk(dir):
            if not dirs:
                folders.append(root)
        return folders

    @staticmethod
    def getlearDirsAndnumbers(dir):
        folders = []
        num_folders = []
        for root, dirs, files in os.walk(dir):
            if root != dir and dirs:
                num_folders.append(len(files))
            if not dirs:
                folders.append(root)
        return folders