#matplotlib inline
from PIL import Image
import numpy as np
import pydicom as dicom
import os
import cv2
import scipy.ndimage
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as ani
import glob
import mcubes
import mudicom

PATIENTS_FOLDER = './Head/data/'
patients = os.listdir(PATIENTS_FOLDER) #listing all directories and files within
patients.sort()

print patients

#------------------------Loading scans and getting numpy array of pixels--------------------

def load_scan(path):
    # InvalidDicomError: File is missing 'DICM' marker. Use force=True to force reading
    # MemoryError
    # slices = [dicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)[400:499]]
    slices = [dicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)]
    # Instance number identifies the sequence of the images in a series
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        # Image position of a patient is the x, y, and z coordinates of the upper left hand corner 
        # (center of the first voxel transmitted) in mm
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        # Slice location is the relative position of the image plane expressed in mm
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    
    image[image == -2000] = 0
    
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(int16)
    return image

# patient = load_scan(PATIENT_FOLDER)
patient = load_scan(PATIENTS_FOLDER + patients[0])
#load_scan returns slices into patient, and pixels are obtained below of that scan
patient_pixels = get_pixels(patient)

# bin width, and color cyan
# plt.hist(patient_pixels.flatten(), bins = 80, color = 'c')
# plt.xlabel('Hounsfield Units (HU)')
# plt.ylabel('Frequency')
# plt.show()

# # show the slice in the middle
# plt.imshow(patient_pixels[60], cmap=plt.cm.gray)
# plt.show()

# mu = plt.imshow(patient_pixels[60], cmap=plt.cm.gray)
# img = mu.image()
# img.numpy()
# img.save_as_plt("dicom.jpeg")

#----------------Functions for resampling and interpolation, and animation------------------


def resample(image, scan, new_spacing): 
    #here image is the pixels of the scan, that is eqt to patient_pixels above
    # plt.imshow(image[60], cmap=plt.cm.gray)
    # plt.show()

    pixel_spacing = []
    for x in scan[0].PixelSpacing:
        pixel_spacing.append(float(x))
    # print pixel_spacing
    # print scan[0].SliceThickness
    spacing = map(float, ([scan[0].SliceThickness] + pixel_spacing))
    spacing = np.array(list(spacing))
    

    #new_spacing is what we want new spacing to be, and calculate resize_factor acc to it
    resize_factor = spacing / new_spacing
    #below we calculate what we want new shape to be based on resize_factor calculated
    #new_real_shape is what we want, not what we have right now
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    #below is just a calculation of resize factors based on shapes
    print new_shape
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    print real_resize_factor
    print new_spacing
    #what above does, is calculates new spacing again. 
    #Since the image changes by real_resize_factor, spacing also does.
    #equivalent to new spacing, but calculated on basis of shapes

    #below we actually change the imagemagick
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    # plt.imshow(image[30], cmap=plt.cm.gray)
    # plt.show()
    
    return image, new_spacing

def animation(patient_pixels, gif_name):
    fig = plt.figure()
    anim = plt.imshow(patient_pixels[0], cmap=plt.cm.gray)
    plt.grid(False)
    def update(i):
        anim.set_array(patient_pixels[i])
        return anim,
    
    a = ani.FuncAnimation(fig, update, frames=range(len(patient_pixels)), interval=50, blit=True)
    # a.save(gif_name, writer='imagemagick')
    #plt.show()

image_resampled, spacing = resample(patient_pixels, patient, [0.25, 1, 1])

print('Shape before resampling\t', patient_pixels.shape)
print('Shape after resampling\t', image_resampled.shape)

# animation(patient_pixels, 'original_patient.gif')
# animation(image_resampled, 'resampled_patient.gif')

print image_resampled.shape[0]
for i in range(image_resampled.shape[0]):
    plt.imshow(image_resampled[i], cmap=plt.cm.gray)
    plt.savefig('Head/output/' + str(i) + '.dcm.png')

#-----------------------Checking if the images are not being replicated---------------------

# img1 = cv2.imread('./Head/output/0.dcm.png',0)
# img2 = cv2.imread('./Head/output/1.dcm.png',0)
# h = img1.shape[0]
# w = img1.shape[1]

# print h, w

# count = 0
# for k in range(0, h):
#     for l in range(0, w):
#         if(img1[k,l] != img2[k,l]):
#             count = count + 1

# print(count)

#------------------------Static filling of ROI and saving new images----------------------

for i in range(image_resampled.shape[0]):
  img = cv2.imread('./Head/output/'+str(i)+'.dcm.png')
  # print(img.shape)
  # print(str(i))
  # print(img)    
  h, w = img.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
  lo =(1,1,1)
  hi = (20,20,20)
  flags = 4
  seed_pt = (255, 255)
  img2 = img.copy()
  cv2.floodFill(img, mask, seed_pt, (255, 0, 0), lo, hi, flags)
  for k in range(1, h):
      for j in range (1, w):
          if(tuple(img[k,j]) == tuple([255,0,0])):
              img[k,j] = img2[k,j]
          else:
              img[k,j] = [0 ,0 ,0]
  cv2.imwrite('./Head/roi_images/'+str(i)+'.png',img)

# for i in range(1501, 1735):
#   image = Image.open('./hola/1/'+str(i)+'.png').convert('RGBA')
#   pixeldata = list(image.getdata())
#   for j,pixel in enumerate(pixeldata):
#       if pixel[:3] == (255,255,255):
#           pixeldata[j] = (255,255,255,0)
#   image.putdata(pixeldata)
#   image.save('./hola/1/'+str(i)+'.png')
# PATIENTS_FOLDER = './hola/1/'
# patients = os.listdir(PATIENTS_FOLDER) #listing all directories and files within
# patients.sort()

#------------------------Running marching cubes on the images with ROI---------------------

X_data = []
files = glob.glob ("./Head/roi_images/*.png")
for myFile in files:
    #print(myFile)
    image = cv2.imread (myFile)
    X_data.append (image)

print('X_data shape:', np.array(X_data).shape)

patient_scans = np.array(X_data)[:,:,:,0]
print patient_scans.shape
print type(patient_scans)
# verts, faces, normals, values = measure.marching_cubes_lewiner(patient_scans, 1)
vertices, triangles = mcubes.marching_cubes(patient_scans, 0)
mcubes.export_mesh(vertices, triangles, "final_image.dae", "MyROI")