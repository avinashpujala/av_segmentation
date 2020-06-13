# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:00:48 2015

@author: pujalaa
"""
import os
import numpy as np
import glob


def get_files(fileDir, pattern=None):
    """
    Returns file names in specified directory, filtered by pattern
    Parameters
    ----------
    fileDir: str
        Path to the file directory
    pattern: str or None
        If str, then filter file names using this regex-style pattern
    Returns
    -------
    fileNames: list
        List of file names in directory
    """
    if pattern is None:
        return np.sort(os.listdir(fileDir))
    else:
        fileNames = glob.fnmatch.filter(os.listdir(fileDir), pattern)
        return fileNames


def subDirs_in_dir(inDir):
    subDirs = [dn for dn in os.listdir(inDir) if os.path.isdir(dn)]
    return subDirs


def recursively_find_paths_with_searchStr(searchDir, searchStr):
    """ Walks down the directory tree for a specified
    search directory and returns the paths to all files or folders
    with specified search string.
    Parameters
    ----------
    searchDir: str
        Path to search directory
    searchStr: str
        Search string to use in looking for files/folders
    Returns
    -------
    paths: list or str
        List of paths returned by the search
    """
    roots, dirs, files = zip(*[out for out in os.walk(searchDir)])
    roots = glob.fnmatch.filter(f'*{searchStr}*.*', roots)
    return np.array(roots)
#
#
# def distribute_files_into_subs(fileDir, n_sub: int = 4, div=750, ext='bmp',
#                           subPrefix='sub'):
#     """ Split files in a directory into 'n_sub' subdirectories within that
#     directory such that the # of files in each subdirectory is divisible by
#     'div'. If the total # of files does not evenly divide into 'div', the does
#     not move the remainder of the files into a subdirectory.
#     Parameters
#     ----------
#     fileDir: str
#         Path to the directory of files to be moved
#     n_sub: int
#         Number of subdirectories into which the files are to be moved
#     div: int or None
#         The # of files in each subfolder wil be divisible by this number
#     ext: str
#         File extension filter
#     subPrefix: str
#         Name prefix of the created subdirectories
#     Returns
#     -------
#     subDirs: List-like
#         Subdirectory paths
#     """
#     import dask
#     if div is None:
#         div = 1
#     filePaths = glob.glob(os.path.join(fileDir, f'*.{ext}'))
#     filePaths = np.array(filePaths)
#     N = len(filePaths)
#     N_div = (N//div)*div
#     inds = np.arange(N_div)
#     subList = sublistsFromList(inds, div)
#     chunkSize = len(subList)//n_sub
#     supList = sublistsFromList(np.arange(len(subList)), chunkSize)
#     if len(supList)>n_sub:
#         supList[-2].extend(supList[-1])
#         supList.pop(-1)
#     inds_sup = []
#     for sl in supList:
#         sub_now = np.array(subList)[sl]
#         inds_=[]
#         for sn in sub_now:
#             inds_.extend(sn)
#         inds_sup.append(inds_)
#
#     subDirs = []
#     for iSub, inds_ in enumerate(inds_sup):
#         sn = f'{subPrefix}_{iSub+1}'
#         dst = os.path.join(fileDir, sn)
#         os.makedirs(dst, exist_ok=True)
#         subDirs.append(dst)
#         print(f'Moving into {sn}, {iSub+1}/{len(inds_sup)}')
#         foo = [dask.delayed(sh.move)(fp, dst) for fp in filePaths[inds_]]
#         dask.compute(*foo)
#     return subDirs
#
#
