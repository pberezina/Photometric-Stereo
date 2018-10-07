import cv2
import numpy as np
import pfmio
import os
import argparse
import errno
from osgeo import gdal


def to_masked_array(img, mask):
    # returns a numpy mask_array type

    m = mask.copy()
    m[m == 0] = 1
    m[m == 255] = 0
    if len(img.shape) >= 3:  # for multidimensional arrays
        result_m = np.ma.zeros(img.shape)
        for z in range(img.shape[2]):
            result_m[:, :, z] = np.ma.masked_array(img[:, :, z], mask=m)
    else:
        result_m = np.ma.masked_array(img, mask=m)

    return result_m


def get_contours(img):
    # trace contours of a mask

    ret, thresh = cv2.threshold(img, 127, 255, 0)  # convert image to binary with 250 threshold val
    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_cnt = img.copy()
    cv2.drawContours(img_cnt, contours, -1, (0, 255, 0), 2)

    # compute coordinates and radius of the circle mask
    (x, y), radius = cv2.minEnclosingCircle(contours[0])

    return x, y, radius


def get_light_direction(img_list, mask, maskX, maskY, maskR):
    # estimates light direction from a metal sphere mask

    light_dir = np.zeros((len(img_list), 3))

    for index, img in enumerate(img_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_masked = to_masked_array(img_gray, mask)    # mask by a metal object mask array

        (_, _, _, maxLoc) = cv2.minMaxLoc(img_masked.filled(0))

        nx = maxLoc[0] - maskX
        ny = maxLoc[1] - maskY
        nz = np.sqrt(maskR ** 2 - nx ** 2 - ny ** 2)
        N = np.divide(np.array([nx, ny, nz]), maskR)
        R = np.array([0, 0, 1.0])
        light_dir[index] = 2 * np.dot(N, R) * N - R

    return light_dir


def get_light_intensity(img_list, mask):

    # calculate light intensity from the brightest pixel on the Lambertian sphere
    light_intensity = np.zeros((len(img_list)))

    for index, img in enumerate(img_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_masked = to_masked_array(img_gray, mask)  # lambertian sphere mask

        (_, maxVal, _, _) = cv2.minMaxLoc(img_masked.filled(0))
        light_intensity[index] = maxVal
    return light_intensity


def calculate_normals_albedo(img_list, mask, light_dir, light_int, mode):
    def lstsq(I, L):
        # inverting the model of reflectance
        Lt = np.transpose(L)
        I = np.transpose(I)
        G = np.dot(np.linalg.inv(np.dot(Lt, L)), np.dot(Lt, I))
        kd = np.linalg.norm(G)
        N = G / kd
        return N, kd

    # set empty arrays
    albedo = to_masked_array(np.ma.zeros((img_list.shape[1], img_list.shape[2])), mask)  # Mask with an object mask
    normals = np.ma.zeros((img_list.shape[1], img_list.shape[2], 3))
    normals[:, :, -1] = 1.0
    normals = to_masked_array(normals, mask)
    I = np.ma.zeros((img_list.shape[0], img_list.shape[1], img_list.shape[2]))

    # calculate normals and albedo for RGB and Grayscale, depending on mode variable
    for i, img in enumerate(img_list):
        if mode == 'GRAY':  # Graydcale
            I[i] = to_masked_array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
        elif mode in [0, 1, 2]:  # RGB
            I[i] = to_masked_array(img, mask)[:, :, mode]

    for i, j in zip(I[0].nonzero()[0], I[0].nonzero()[1]):
        # calibrate pixel intensity with light intensity from the lambertian sphere
        normals[i, j], albedo[i, j] = lstsq(light_int * I[:, i, j], light_dir)

    return normals, albedo / albedo.max()


def main(args):
    out_path = os.path.join(args.input_path, 'results')
    try:
        os.makedirs(out_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            pass

    img_list = []
    mask_obj = cv2.imread(os.path.join(args.input_path, 'mask.png'), cv2.IMREAD_GRAYSCALE)
    mask_metal = cv2.imread(os.path.join(args.input_path, 'mask_dir_1.png'), cv2.IMREAD_GRAYSCALE)
    mask_lambert = cv2.imread(os.path.join(args.input_path, 'mask_I.png'), cv2.IMREAD_GRAYSCALE)

    try:       # load pbm files
        for file in os.listdir(args.input_path):
            if file.endswith('pbm'):
                with open(os.path.join(args.input_path, file), encoding="utf8", errors='ignore') as ffile:
                    pfm_arr = pfmio.load_pfm(ffile)
                    pfm_arr[np.isnan(pfm_arr)] = 0
                    img_list.append(pfm_arr)
    except:
        try:   # load tif files
            for file in os.listdir(args.input_path):
                if file.endswith('tif'):
                    file_path = os.path.join(args.input_path, file)
                    im_array = np.array(gdal.Open(file_path).ReadAsArray())
                    im_array_flipped = np.moveaxis(im_array, 0, -1)  # put 3d axis last to match pfm_arr
                    img_list.append(im_array_flipped)
        except:
            print("Could not load images")
            exit(1)

    img_list = np.asarray(img_list)

    light_direction = get_light_direction(img_list, mask_metal, *get_contours(mask_metal))   # light directions
    light_intensity = get_light_intensity(img_list, mask_lambert)

    # obtain normals using gray scale image
    normals, albedo_gray = calculate_normals_albedo(img_list, mask_obj, light_direction, light_intensity, 'GRAY')
    # scale x,y,z components of obtained normals in RGB channels from 0 to 1
    normals_scaled = to_masked_array(np.divide(normals.copy() + 1, 2), mask_obj).filled(0)

    # obtain albedo for each RGB channel and store in .png
    albedos_rgb = np.ma.zeros((img_list.shape[1], img_list.shape[2], 3))
    for color in [0,1,2]:         # 0 Red, 1 Green, 2 Blue as in original images
        _, albedos_rgb[:,:,color] = calculate_normals_albedo(img_list, mask_obj, light_direction, light_intensity, color)
        cv2.imwrite(os.path.join(out_path, 'Albedo_channel{}.png'.format(color)), albedos_rgb[:, :, color] * 255)

    # save gray-scale albedo and normals to .png
    # convert from 0-1 float to 0-255. Also opencv uses BGR, so we tweak things around by replacing channels
    cv2.imwrite(os.path.join(out_path, 'color.png'), albedos_rgb[:,:,::-1]*255)
    cv2.imwrite(os.path.join(out_path, 'normals.png'), normals_scaled * 255)
    cv2.imwrite(os.path.join(out_path, 'Albedo_gray.png'), albedo_gray * 255)

    # re-rendered picture with recovered normal and albedo under illumination dir, the same as the viewing direction
    rerendered =  albedo_gray * np.dot(normals, np.array([0,0,1]))
    cv2.imwrite(os.path.join(out_path, 'rerendered.png'), rerendered * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    main(args)