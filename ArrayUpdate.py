#combined array with google drive
import numpy as np
import cv2
import time
import itertools
import csv
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from collections import Counter

gauth = GoogleAuth()

def updateArray(array,idArray, chipToUpdate, idType):
    array.append(chipToUpdate)
    idArray.append(idType)
    return array, idArray

def transpose( array, idArray, keyDictionary):

    finalCountArray = []

    fileTime = time.strftime("%Y%m%d-%H%M%S")#generate current time
    #Try to load saved client credentials
    gauth.LoadCredentialsFile("credentials.json")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("credentials.json")

    np.transpose(idArray)
    np.transpose(array)

    #convert dictionary to array

    writeKeyArray = []
    for key, items in keyDictionary.items():
        finalCountArray.append(idArray.count(items))
        writeKeyArray.append(str( "Total of ") + items)
    # print(finalCountArray)

    combine = np.stack((idArray,array), axis=-1)

    finalCombine = np.stack((writeKeyArray, finalCountArray), axis=-1)

    print(finalCombine)

    with open(fileTime + '.csv', 'w', newline='') as f:
            a = csv.writer(f, delimiter=',')
            initial=[['ID','Type']]
            a.writerows(initial)
            a.writerows(combine)

            a.writerows(finalCombine)

    print(combine)

    # GDrive operation
    drive = GoogleDrive(gauth)
    file5 = drive.CreateFile()
    # Read file and set it as a content of this instance.
    file5.SetContentFile(fileTime + '.csv')
    file5.Upload()  # Upload the file.
    print("success")
    # title: cat.png, mimeType: image/png

