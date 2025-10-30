import os
import ctypes
from ctypes import *
from array import *
import csv
import numpy as np
from PIL import Image, ImageDraw
import GoSdk_MsgHandler
import uuid
import cv2
import time
from datetime import datetime
import matplotlib.pyplot as plt
import math
## import open3d as o3d - uncomment if issues
try:
    import open3d as o3d
except ImportError:
    # skip Open3D (we don’t need visualization here)
    o3d = None

### Load Api
# Please define your System Environment Variable as GO_SDK_4. It should reference the root directory of the SDK package.
SdkPath = os.environ['GO_SDK_4']

#Linux
kApi = ctypes.cdll.LoadLibrary(os.path.join(SdkPath, 'lib', 'linux_arm64', 'libkApi.so'))
GoSdk = ctypes.cdll.LoadLibrary(os.path.join(SdkPath, 'lib', 'linux_arm64', 'libGoSdk.so'))

print("[INFO] SDK libraries loaded successfully.")

### Constant Declaration and Instantiation
kNULL = 0
kTRUE = 1
kFALSE = 0
kOK = 1
GO_DATA_MESSAGE_TYPE_SURFACE_POINT_CLOUD = 28
GO_DATA_MESSAGE_TYPE_MEASUREMENT = 10
GO_DATA_MESSAGE_TYPE_SURFACE_INTENSITY = 9
GO_DATA_MESSAGE_TYPE_UNIFORM_SURFACE = 8
GO_DATA_MESSAGE_TYPE_UNIFORM_PROFILE = 7
GO_DATA_MESSAGE_TYPE_PROFILE_POINT_CLOUD = 5
GO_DATA_MESSAGE_TYPE_STAMP = 0
RECEIVE_TIMEOUT = 10000

### Gocator DataType Declarations
kObject = ctypes.c_void_p
kValue = ctypes.c_uint32
kSize = ctypes.c_ulonglong
kAssembly = ctypes.c_void_p
GoSystem = ctypes.c_void_p
GoSensor = ctypes.c_void_p
GoDataSet = ctypes.c_void_p
GoDataMsg = ctypes.c_void_p
kChar = ctypes.c_byte
kBool = ctypes.c_bool
kCall = ctypes.c_bool
kCount = ctypes.c_uint32

class GoStampData(Structure):
    _fields_ = [("frameIndex", c_uint64), ("timestamp",c_uint64), ("encoder", c_int64), ("encoderAtZ", c_int64), ("status", c_uint64), ("id", c_uint32)]

class GoMeasurementData(Structure):
    _fields_ = [("numericVal", c_double), ("decision", c_uint8), ("decisionCode", c_uint8)]

class kIpAddress(Structure):
    _fields_ = [("kIpVersion", c_int32),("kByte",c_char*16)]

### Define Argtype and Restype
GoSdk.GoDataSet_At_argtypes = [kObject, kSize]
GoSdk.GoDataSet_At.restype = kObject
GoSdk.GoDataMsg_Type.argtypes = [kObject]
GoSdk.GoDataMsg_Type.restype = kValue
GoSdk.GoSurfaceMsg_RowAt.restype = c_int64
GoSdk.GoUniformSurfaceMsg_RowAt.restype = ctypes.POINTER(ctypes.c_int16)
GoSdk.GoSurfaceIntensityMsg_RowAt.restype = ctypes.POINTER(ctypes.c_uint8)
GoSdk.GoSurfacePointCloudMsg_RowAt.restype = ctypes.POINTER(ctypes.c_int16)
GoSdk.GoStampMsg_At.restype = ctypes.POINTER(GoStampData)
GoSdk.GoMeasurementMsg_At.restype = ctypes.POINTER(GoMeasurementData)
GoSdk.GoResampledProfileMsg_At.restype = ctypes.POINTER(ctypes.c_short)
GoSdk.GoSensor_CopyFile.argtypes = [GoSensor ,c_char_p, c_char_p]
GoSdk.GoSensor_FileNameAt.argtypes = [GoSensor,ctypes.c_uint64,c_char_p,ctypes.c_uint64]
GoSdk.GoProfileMsg_At.restype = ctypes.POINTER(ctypes.c_int16)


def getVersionStr():
    version = ctypes.create_string_buffer(32)
    myVersion = GoSdk.GoSdk_Version()
    kApi.kVersion_Format(myVersion, version, 32)
    return str(ctypes.string_at(version))

def kObject_Destroy(object):
    if (object != kNULL):
        kApi.xkObject_DestroyImpl(object, kFALSE)

def save_image(image, measurement_type, save_dir, format):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{measurement_type}_{timestamp}.{format}"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Image saved at: {filepath}")

def save_pcd(data_3DXYZ, measurement_type, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{measurement_type}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # # --- Save CSV ---
    # with open(filepath, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow(["X", "Y", "Z"])
    #     writer.writerows(data_3DXYZ)

    # print(f"CSV file saved at: {filepath}")


    # --- Save PLY ---
    ply_path = os.path.join(save_dir, filename + ".ply")
    if o3d is not None:
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            
            # Filter out NaN values for better PLY files
            valid_mask = ~np.isnan(data_3DXYZ).any(axis=1)
            valid_points = data_3DXYZ[valid_mask]
            
            if len(valid_points) == 0:
                print(f"[WARN] No valid points found for PLY file: {ply_path}")
                return
            
            pcd.points = o3d.utility.Vector3dVector(valid_points)
            
            # Add basic colors (white) to make visualization easier
            colors = np.ones((len(valid_points), 3))  # White color for all points
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save PLY file
            success = o3d.io.write_point_cloud(ply_path, pcd)
            
            if success:
                print(f"PLY file saved at: {ply_path}")
                print(f"  - Valid points: {len(valid_points):,}")
                print(f"  - Filtered out: {len(data_3DXYZ) - len(valid_points):,} NaN points")
                print(f"  - File size: {os.path.getsize(ply_path) / 1024:.1f} KB")
            else:
                print(f"[ERROR] Failed to save PLY file: {ply_path}")
                
        except Exception as e:
            print(f"[ERROR] PLY file creation failed: {e}")
            print(f"[ERROR] Data shape: {data_3DXYZ.shape}, NaN count: {np.isnan(data_3DXYZ).sum()}")
    else:
        print(f"[WARN] Open3D not available, skipping PLY file creation: {ply_path}")


def compute_surface_plane_tilt(x_angle_deg, y_angle_deg):
   
    # Convert to radians
    x_rad = math.radians(x_angle_deg)
    y_rad = math.radians(y_angle_deg)

    # Compute tilt in radians using the formula: tilt = acos(cos(x) * cos(y))
    tilt_rad = math.acos(math.cos(x_rad) * math.cos(y_rad))

    # Convert back to degrees
    tilt_deg = math.degrees(tilt_rad)

    print(f"Tilt Angle: {tilt_deg:.3f} degrees")


measurement_decision = None


def RecieveData(dataset):

    global measurement_decision
    XAngle = None
    YAngle = None

    ## loop through all items in result message
    for i in range(GoSdk.GoDataSet_Count(dataset)):
        k_object_address = GoSdk.GoDataSet_At(dataset, i)
        dataObj = GoDataMsg(k_object_address)

        ## Retrieve stamp message
        if GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_STAMP:
            stampMsg = dataObj
            msgCount = GoSdk.GoStampMsg_Count(stampMsg)
            
            for k in range(msgCount):
                stampDataPtr = GoSdk.GoStampMsg_At(stampMsg,k)
                stampData = stampDataPtr.contents
                # print("frame index: ", stampData.frameIndex)
                # print("time stamp: ", stampData.timestamp)
                # print("encoder: ", stampData.encoder)
                # print("sensor ID: ", stampData.id)
                # print()

        if GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_MEASUREMENT:
            measurementMsg = dataObj
            msgCount = GoSdk.GoMeasurementMsg_Count(measurementMsg)
            print("Measurement Message batch count: %d" % msgCount);

            for k in range(GoSdk.GoMeasurementMsg_Count(measurementMsg)):
                measurementDataPtr = (GoSdk.GoMeasurementMsg_At(measurementMsg, k))
                measurementData = measurementDataPtr.contents #(measurementDataPtr, POINTER(GoMeasurementData)).contents
                # print(measurementData.__class__._fields_)

                measurementID = GoSdk.GoMeasurementMsg_Id(measurementMsg)
                #measurementName = GoSdk.GoMeasurement_Name(measurementMsg)
                measurement_decision = measurementData.decision if measurementID == 1 else 1
                measurement_unit = "mm" if measurementID == 1 else "degrees"              

                if measurementID == 3: XAngle = measurementData.numericVal
                elif measurementID == 4: YAngle = measurementData.numericVal

                print("Measurement ID: ", measurementID)
                #print("Measurement Name: ", measurementName)
                print(f"Measurement Value: { measurementData.numericVal} {measurement_unit}")
                print("Measurment Decision: " + str(measurement_decision))
                print()
            
                if XAngle is not None and YAngle is not None:
                    compute_surface_plane_tilt(XAngle, YAngle)
                
        if GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_UNIFORM_SURFACE:
            surfaceMsg = dataObj
            print("Surface Message")

            #resolutions and offsets (cast to mm)
            XResolution = float((GoSdk.GoUniformSurfaceMsg_XResolution(surfaceMsg)))/1000000.0
            YResolution = float((GoSdk.GoUniformSurfaceMsg_YResolution(surfaceMsg)))/1000000.0
            ZResolution = float((GoSdk.GoUniformSurfaceMsg_ZResolution(surfaceMsg)))/1000000.0
            XOffset = float((GoSdk.GoUniformSurfaceMsg_XOffset(surfaceMsg)))/1000.0
            YOffset = float((GoSdk.GoUniformSurfaceMsg_YOffset(surfaceMsg)))/1000.0
            ZOffset = float((GoSdk.GoUniformSurfaceMsg_ZOffset(surfaceMsg)))/1000.0
            width = GoSdk.GoUniformSurfaceMsg_Width(surfaceMsg)
            length = GoSdk.GoUniformSurfaceMsg_Length(surfaceMsg)
            size = width * length

            print("Surface data width: " + str(width))
            print("Surface data length: " + str(length))
            print("Total num points: " + str(size)) 

            #Generate Z points
            surfaceDataPtr = GoSdk.GoUniformSurfaceMsg_RowAt(surfaceMsg, 0)
            Z = np.ctypeslib.as_array(surfaceDataPtr, shape=(size,))  
            Z = Z.astype(np.double)
            #remove -32768 and replace with nan
            Z[Z==-32768] = np.nan    
            #scale to real world units (for Z only)
            Z = (Z * ZResolution) + ZOffset     

            #generate X points
            X = (np.asarray(range(width), dtype=np.double) * XResolution) + XOffset
            X = np.tile(X, length)

            #generate Y points
            Y = (np.arange(length, dtype=np.double)* YResolution) + YOffset
            Y = np.repeat(Y, repeats=width)

            #Generate X, Y, Z array for saving
            data_3DXYZ = np.stack((X,Y,Z), axis = 1)


            # #write to file as np array (fast)
            # start = time.time()
            # unique_filename = str(uuid.uuid4())
            # np.save(unique_filename+"XYZ"+".npy",data_3DXYZ)
            # print("wrote to file "+unique_filename+ "XYZ" + ".npy")
            # print("Save npy file time: ",time.time() - start)

            #save PLY
            save_pcd(data_3DXYZ, "UNIFORM_SURFACE", "UNIFORM_SURFACE_PLY")
            print()

        #     #--------------------------------------------------------------

        #     Z = data_3DXYZ[:, 2]
        #     Z_mean = np.nanmean(Z)  # Mean Z
        #     Z_cleaned = np.where(np.isnan(Z), Z_mean, Z)

        #     # Normalize the array to the range [0, 255] and convert to uint8
        #     normalized_data = (Z_cleaned - np.min(Z_cleaned)) / (np.max(Z_cleaned) - np.min(Z_cleaned)) * 255
        #     array_uint8 = normalized_data.astype(np.uint8)

        #     # Reshape the 1D array into a 2D array with dimensions (length, width)
        #     array_2d = array_uint8.reshape((length, width))  # Ensure correct dimensions

        #     # Convert to an image and save as BMP
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #     img = Image.fromarray(array_2d, 'L')  # Mode 'L' ensures it's grayscale
        #     #img = Image.fromarray(color_image)
        #     img.save(f'UNIFORM_SURFACE_IMG/US_{timestamp}.bmp')

        #     #--------------------------------------------------------------

        #     image = np.reshape(Z, (-1, width))

        #     # Handle NaN values before normalization
        #     if np.isnan(image).any():
        #         mean_val = np.nanmean(image)  # Compute mean excluding NaNs
        #         image = np.nan_to_num(image, nan=mean_val)  # Replace NaNs with mean

        #     maxval = np.max(image) if np.max(image) > 0 else 1  # Prevent division by zero
        #     image = (image / maxval) * 255.0

        #     # Ensure valid range
        #     image = np.clip(image, 0, 255).astype(np.uint8)

        #     # Resize with cubic interpolation
        #     image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        #     # Display the image
        #     #cv2.imshow("UNIFORM_SURFACE", image)
            
        #     # Save the processed image
        #     save_image(image, "UNIFORM_SURFACE", "UNIFORM_SURFACE_IMG", "png")

        #     # Skip GUI-related wait that causes errors on headless systems
        #     try:
        #         # cv2.waitKey(500)  # Not supported on headless systems
        #         pass
        #     except Exception as e:
        #         print(f"[!] Skipping cv2.waitKey due to GUI error: {e}")

        #     time.sleep(1)
        #     print()

        # elif GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_SURFACE_INTENSITY:
        #     print("Intensity Message")
        #     surfaceIntensityMsg = dataObj

        #     #resolutions and offsets (cast to mm)
        #     XResolution = float((GoSdk.GoSurfaceIntensityMsg_XResolution(surfaceIntensityMsg)))/1000000.0
        #     YResolution = float((GoSdk.GoSurfaceIntensityMsg_YResolution(surfaceIntensityMsg)))/1000000.0
        #     XOffset = float((GoSdk.GoSurfaceIntensityMsg_XOffset(surfaceIntensityMsg)))/1000.0
        #     YOffset = float((GoSdk.GoSurfaceIntensityMsg_YOffset(surfaceIntensityMsg)))/1000.0
        #     width = GoSdk.GoSurfaceIntensityMsg_Width(surfaceIntensityMsg)
        #     length = GoSdk.GoSurfaceIntensityMsg_Length(surfaceIntensityMsg)
        #     size = width * length
            
        #     print("Surface data width: " + str(width))
        #     print("Surface data length: " + str(length))
        #     print("Total num points: " + str(size)) 

        #     #Generate I points
        #     surfaceIntensityDataPtr = GoSdk.GoSurfaceIntensityMsg_RowAt(surfaceIntensityMsg, 0)
        #     I = np.array((surfaceIntensityDataPtr[0:width*length]), dtype=np.uint8)
            
        #     #generate X points
        #     X = (np.asarray(range(width), dtype=np.double) * XResolution) + XOffset
        #     X = np.tile(X, length)

        #     #generate Y points
        #     #
        #     Y = (np.arange(length, dtype=np.double)* YResolution) + YOffset
        #     Y = np.repeat(Y, repeats=width)
        #     #print("Y array generation time: ",time.time() - start)

        #     #Generate X, Y, Z array for saving
        #     data_3DXYI = np.stack((X,Y,I), axis = 1)                

        #     # #write to file as np array (fast)
        #     # start = time.time()
        #     # unique_filename = str(uuid.uuid4())
        #     # np.save(unique_filename+"XYI"+".npy",data_3DXYI)
        #     # print("wrote to file "+unique_filename+ "XYI" + ".npy")
        #     # print("Save npy file time: ",time.time() - start)
            
        #     ##write to CSV (slow)       
        #     save_pcd(data_3DXYI, "SURFACE_INTENSITY", "SURFACE_INTENSITY_CSV")


        #     #Save BMP Image
        #     #Normalize the array to the range [0, 255] and convert to uint8
        #     normalized_data = (I - np.min(I)) / (np.max(I) - np.min(I)) * 255
        #     array_uint8 = normalized_data.astype(np.uint8)

        #     # Reshape the 1D array into a 2D array with dimensions (length, width)
        #     array_2d = array_uint8.reshape((length, width))  # Ensure correct dimensions

        #     # I_norm = (I - np.min(I)) / (np.max(I) - np.min(I)) 
        #     # # Reshape to 2D
        #     # array_2d = I_norm.reshape((length, width))
        #     # colormap = plt.cm.jet(array_2d)  # Returns an RGBA image
        #     # color_image = (colormap[:, :, :3] * 255).astype(np.uint8)

        #     # Convert to an image and save as BMP
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #     img = Image.fromarray(array_2d, 'L')  # Mode 'L' ensures it's grayscale
        #     #img = Image.fromarray(color_image)
        #     img.save(f'SURFACE_INTENSITY_IMG/SI_{timestamp}.bmp')

        #     ## Save PNG Image
        #     #Display the surface (it look square unless a perspective correction is done)
        #     image = np.reshape(I, (-1, width))
        #     maxval = np.nanmax(image)
        #     image = (image / maxval) * 255.0
        #     image = image.astype(np.uint8) 
        #     image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)         
        #     #cv2.imshow("SURFACE_INTENSITY", image)
        #     save_image(image, "SURFACE_INTENSITY", "SURFACE_INTENSITY_IMG","png")
        #     cv2.waitKey(500)
        #     print()

        # elif GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_UNIFORM_PROFILE:
        #     print("Profile Message")
        #     profileMsg = dataObj
        #     k=0
        #     for k in range(GoSdk.GoResampledProfileMsg_Count(profileMsg)):
        #         #resolutions and offsets (cast to mm)
        #         XResolution = float((GoSdk.GoResampledProfileMsg_XResolution(profileMsg)))/1000000.0
        #         ZResolution = float((GoSdk.GoResampledProfileMsg_ZResolution(profileMsg)))/1000000.0
        #         XOffset = float((GoSdk.GoResampledProfileMsg_XOffset(profileMsg)))/1000.0
        #         ZOffset = float((GoSdk.GoResampledProfileMsg_ZOffset(profileMsg)))/1000.0
        #         width = GoSdk.GoProfileMsg_Width(profileMsg)
        #         size = width

        #         #Generate Z points
        #         start = time.time()
        #         profileDataPtr = GoSdk.GoResampledProfileMsg_At(profileMsg, k)
        #         Z = np.ctypeslib.as_array(profileDataPtr, shape=(size,))
        #         Z = Z.astype(np.double)
        #         Z[Z==-32768] = np.nan                 
        #         Z = (Z * ZResolution) + ZOffset     
        #         print("Z array generation time: ",time.time() - start)

        #         #generate X points
        #         start = time.time()
        #         X = (np.asarray(range(width), dtype=np.double) * XResolution) + XOffset
        #         print("X array generation time: ",time.time() - start)

        #         #Generate X, Y, Z array for saving
        #         data_3DXZ = np.stack((X,Z), axis = 1)   

        #         #write to file as np array (fast)
        #         #unique_filename = str(uuid.uuid4())
        #         #np.save(unique_filename+"XZ"+".npy",data_3DXZ)
                
        #         #write to CSV (slow)
        #         #unique_filename = str(uuid.uuid4())
        #         #with open(unique_filename+"XZ.csv",'w',newline='') as csvfile:
        #         #    writer = csv.writer(csvfile,delimiter=',')
        #         #    writer.writerow(["X","Z"])
        #         #    writer.writerows(data_3DXZ)
        #         #print("wrote to file "+unique_filename)
        #         print()

        # elif GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_PROFILE_POINT_CLOUD:
        #     print("Non-uniform Profile Message")
        #     profileMsg = dataObj
        #     k=0
        #     for k in range(GoSdk.GoResampledProfileMsg_Count(profileMsg)):
        #         #resolutions and offsets (cast to mm)
        #         XResolution = float((GoSdk.GoResampledProfileMsg_XResolution(profileMsg)))/1000000.0
        #         ZResolution = float((GoSdk.GoResampledProfileMsg_ZResolution(profileMsg)))/1000000.0
        #         XOffset = float((GoSdk.GoResampledProfileMsg_XOffset(profileMsg)))/1000.0
        #         ZOffset = float((GoSdk.GoResampledProfileMsg_ZOffset(profileMsg)))/1000.0
        #         width = GoSdk.GoProfileMsg_Width(profileMsg)
        #         size = width
        #         dataLength = size*2 #each data point contains an X and Z component

        #         #Generate Z points
        #         start = time.time()
        #         profileDataPtr = GoSdk.GoProfileMsg_At(profileMsg, k)
        #         XZ = np.ctypeslib.as_array(profileDataPtr, shape=(dataLength,))
        #         XZ = XZ.astype(np.double)
        #         XZ[XZ==-32768] = np.nan
        #         print("Size = "+ str(dataLength))
        #         print("XZ arr:")
        #         print(XZ)
        #         X = XZ[0::2]
        #         Z = XZ[1::2]       
        #         X = (X * XResolution) + XOffset               
        #         Z = (Z * ZResolution) + ZOffset     
        #         print("X & Z array generation time: ",time.time() - start)

        #         #Generate X, Z array for saving
        #         data_3DXZ = np.stack((X,Z), axis = 1)   

        #         #write to file as np array (fast)
        #         unique_filename = 'point_clouds/'+ str(uuid.uuid4())
        #         np.save(unique_filename+"XZ"+".npy",data_3DXZ)
                
        #         #write to CSV (slow)
        #         unique_filename = 'point_clouds/'+ str(uuid.uuid4())
        #         with open(unique_filename+"XZ.csv",'w',newline='') as csvfile:
        #             writer = csv.writer(csvfile,delimiter=',')
        #             writer.writerow(["X","Z"])
        #             writer.writerows(data_3DXZ)
        #         print("wrote to file "+unique_filename)
                
        #         print()

        # elif GoSdk.GoDataMsg_Type(dataObj) == GO_DATA_MESSAGE_TYPE_SURFACE_POINT_CLOUD:
        #     surfaceMsg = dataObj
        #     print("Non-uniform Surface Message")
        #     #resolutions and offsets (cast to mm)
        #     XResolution = float((GoSdk.GoSurfacePointCloudMsg_XResolution(surfaceMsg)))/1000000.0
        #     YResolution = float((GoSdk.GoSurfacePointCloudMsg_YResolution(surfaceMsg)))/1000000.0
        #     ZResolution = float((GoSdk.GoSurfacePointCloudMsg_ZResolution(surfaceMsg)))/1000000.0
        #     XOffset = float((GoSdk.GoSurfacePointCloudMsg_XOffset(surfaceMsg)))/1000.0
        #     YOffset = float((GoSdk.GoSurfacePointCloudMsg_YOffset(surfaceMsg)))/1000.0
        #     ZOffset = float((GoSdk.GoSurfacePointCloudMsg_ZOffset(surfaceMsg)))/1000.0
        #     width = GoSdk.GoSurfacePointCloudMsg_Width(surfaceMsg)
        #     length = GoSdk.GoSurfacePointCloudMsg_Length(surfaceMsg)
        #     size = width * length
        #     dataLength = size*3 #each data point has an X, Y, Z component

    
        #     print("Surface data width: " + str(width))
        #     print("Surface data length: " + str(length))
        #     print("Total num points (X,Y,Z): " + str(dataLength))

        
        #     #Generate Z points
        #     start = time.time()
        #     surfaceDataPtr = GoSdk.GoSurfacePointCloudMsg_RowAt(surfaceMsg, 0)
        #     XYZ = np.ctypeslib.as_array(surfaceDataPtr, shape=(dataLength,))  
        #     XYZ = XYZ.astype(np.double)
        #     #remove -32768 and replace with nan
        #     XYZ[XYZ==-32768] = np.nan    
        #     #break into X, Y, Z lists
        #     X = XYZ[0::3]
        #     Y = XYZ[1::3]
        #     Z = XYZ[2::3]

        #     #scale to real world units (for Z only)                  
        #     Z = (Z * ZResolution) + ZOffset    
        #     print("Z array generation time: ",time.time() - start)

    
        #     #generate X points
        #     start = time.time()
        #     X = (X * XResolution) + XOffset
        #     #X = np.tile(X, length)
        #     print("X array generation time: ",time.time() - start)

    
        #     #generate Y points
        #     start = time.time()
        #     Y = (Y* YResolution) + YOffset
        #     #Y = np.repeat(Y, repeats=width)
        #     print("Y array generation time: ",time.time() - start)

    
        #     #Generate X, Y, Z array for saving
        #     print(X.size)
        #     print(Y.size)
        #     print(Z.size)
        #     data_3DXYZ = np.stack((X,Y,Z), axis = 1)
        #     print(data_3DXYZ)

        #     #write to file as np array (fast)
        #     start = time.time()
        #     unique_filename = str(uuid.uuid4())
        #     np.save(unique_filename+"XYZ"+".npy",data_3DXYZ)
        #     print("wrote to file "+unique_filename+ "XYZ" + ".npy")
        #     print("Save npy file time: ",time.time() - start)
            
        #     #write to CSV (slow)
        #     start = time.time()
        #     unique_filename = str(uuid.uuid4())
        #     with open(unique_filename+"XYZ.csv",'w',newline='') as csvfile:
        #       writer = csv.writer(csvfile,delimiter=',')
        #       writer.writerow(["X","Y","Z"])
        #       writer.writerows(data_3DXYZ)
        #     print("wrote to file "+unique_filename + "XYZ.csv")
        #     print("Save CSV file time: ",time.time() - start)        


    kObject_Destroy(dataset)

def get_measurement_decision():
    return measurement_decision

if __name__ == "__main__":
    # Instantiate system objects
    api = kAssembly(kNULL)
    system = GoSystem(kNULL)
    sensor = GoSensor(kNULL)
    dataset = GoDataSet(kNULL)
    dataObj = GoDataMsg(kNULL)
    changed = kBool(kNULL)

    print('Sdk Version is: ' + getVersionStr())

    GoSdk.GoSdk_Construct(byref(api))  # Build API
    GoSdk.GoSystem_Construct(byref(system), kNULL)  # Construct sensor system

    #connect to sensor via IP
    sensor_IP = b"192.168.1.10" #default for local emulator is 127.0.0.1
    ipAddr_ref = kIpAddress()
    kApi.kIpAddress_Parse(byref(ipAddr_ref), sensor_IP)
    GoSdk.GoSystem_FindSensorByIpAddress(system,byref(ipAddr_ref),byref(sensor))


    GoSdk.GoSensor_Connect(sensor)  # Connect to the sensor
    GoSdk.GoSystem_EnableData(system, kTRUE)  # Enable the sensor's data channel to receive measurement data
    #GoSdk.GoSensor_Start(sensor)  # Start the sensor to gather data
    print(f"Connected to scanner via {sensor_IP}!")

    #show files on the sensor
    #capacity = 64
    #buffer = ctypes.create_string_buffer(capacity)
    #generate a list of jobs
    #for fileIndex in range((GoSdk.GoSensor_FileCount(sensor))):
    #    GoSdk.GoSensor_FileNameAt(sensor,fileIndex,buffer,capacity)
    #    print(buffer.value)

    #Change job file, replace requestedjob.job with the job needed to be loaded
    #GoSdk.GoSensor_CopyFile(sensor,b'requestedjob.job',b'_live.job')


    #Initialize message handler manager
    Mgr = GoSdk_MsgHandler.MsgManager(GoSdk, system, dataset)

    #Set data handler which spawns a worker thread to recieve input data
    Mgr.SetDataHandler(RECEIVE_TIMEOUT, RecieveData)

    #Issue a stop then start incase the emulator is still running. For live sensors, only a start is needed.
    GoSdk.GoSensor_Stop(sensor) 
    GoSdk.GoSensor_Snapshot(sensor) 
    
    #Do nothing
    while(input() != "exit"):
        pass
    
    #Can close thread manually by recalling data handler with kNull passed
    Mgr.SetDataHandler(GoSdk, system, dataset, RECEIVE_TIMEOUT, kNULL)


    ### Destroy the system object and api
    kObject_Destroy(system)
    kObject_Destroy(api)