# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:00:48 2015

@author: pujalaa
"""
import os
import numpy as np
import glob


def createOrAppendToHdf(hFile, keyName, arr, verbose: bool = False):
    """
    Create a dataset (can be within in group) within a given HDF file
    if the dataset with specified name doesn't exist else append to the
    existing dataset
    Parameters
    ----------
    hFile: HDF file object
        HDF file object to create or append dataset in.
    keyName: str
        Path to the dataset in the HDF file. For e.g., 'foo' or 'foo/bar'.
    arr: array
        Array to write to the dataset.
    verbose: bool
        If True, prints name of the dataset key
    Returns
    -------
    hFile: HDF file object
        HDF file object with dataset.
    """
    if keyName not in hFile:
        if verbose:
            print(f'Creating {keyName} in hdf file')
        hFile.create_dataset(keyName, data=arr, maxshape=(None, *arr.shape[1:]),
                             compression='lzf')
    else:
        if verbose:
            print(f'Appending to {keyName} in h5 file')
        hFile[keyName].resize((hFile[keyName].shape[0] + arr.shape[0]), axis=0)
        hFile[keyName][-arr.shape[0]:] = arr
    return hFile


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


def sublists_from_list(input_list, chunk_size):
    """
    Given a list, chunks it into sizes specified and returns the chunks as items
    in a new list
    Parameters
    ----------
    input_list: list
        Original list
    chunk_size: int
        Size of sublists
    Returns
    -------
    sub_lists: list
        Sublists
    """

    subList, supList = [], []
    for itemNum, item in enumerate(input_list):
        if np.mod(itemNum+1, chunk_size) == 0:
            subList.append(item)
            supList.append(subList)
            subList = []
        else:
            subList.append(item)
    supList.append(subList)
    supList = list(filter(lambda x: len(x) != 0, supList))  # Remove zero-length lists
    return supList


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
