from __future__ import print_function
from collections import OrderedDict
from blaze.compute import pandas
import numpy as np
import os
from skimage import io, morphology, measure


# Read image, analyze, find region properties
from toolz.sandbox.parallel import fold


def get_image_region_list(filename):
    # Read image file
    image = io.imread(filename, as_grey=True)

    # Thresholding
    image_threshold = np.where(image > np.mean(image), 0., 1.0)

    # Dilation
    size_neighborhood = 4
    image_dilated = morphology.dilation(image_threshold, np.ones((size_neighborhood, size_neighborhood)))

    # Label regions
    label_list = measure.label(image_dilated) + 1

    # Create label list
    label_list = (image_threshold * label_list).astype(int)

    # Region properties
    region_list = measure.regionprops(label_list)

    return region_list


# Find the region with the largest area
def get_max_area(filename):
    region_list = get_image_region_list(filename)

    max_area = None
    region_count = 0
    for prop in region_list:
        region_count += 1
        if max_area is None:
            max_area = prop
        else:
            if prop.area > max_area.area:
                max_area = prop
    return max_area, region_count


def get_max_area_dict(file_name, folder=None):
    prop, region_count = get_max_area(file_name)

    max_area_dict = OrderedDict((('class', folder),
                                 ('file_name', os.path.basename(file_name)),
                                 ('reg_count', region_count),
                                 # ('label', prop.label),
                                 ('centroid_row', prop.centroid[0]),  # 0D:  location
                                 ('centroid_col', prop.centroid[1]),
                                 ('diameter_equivalent', prop.equivalent_diameter),  # 1D
                                 ('length_minor_axis', prop.minor_axis_length),
                                 ('length_major_axis', prop.major_axis_length),
                                 ('ratio_eccentricity', prop.eccentricity),
                                 ('perimeter', prop.perimeter),
                                 # ('orientation', prop.orientation),  # ranges from -pi/2 to pi/2
                                 ('area', prop.area),  # 2D
                                 ('area_convex', prop.convex_area),
                                 ('area_filled', prop.filled_area),
                                 ('box_min_row', prop.bbox[0]),
                                 ('box_max_row', prop.bbox[2]),
                                 ('box_min_col', prop.bbox[1]),
                                 ('box_max_col', prop.bbox[3]),
                                 ('ratio_extent', prop.extent),
                                 ('ratio_solidity', prop.solidity),
                                 ('inertia_tensor_eigenvalue1', prop.inertia_tensor_eigvals[0]),
                                 ('inertia_tensor_eigenvalue2', prop.inertia_tensor_eigvals[1]),
                                 ('moments_hu1', prop.moments_hu[0]),  # translation, scale and rotation invariant
                                 ('moments_hu2', prop.moments_hu[1]),
                                 ('moments_hu3', prop.moments_hu[2]),
                                 ('moments_hu4', prop.moments_hu[3]),
                                 ('moments_hu5', prop.moments_hu[4]),
                                 ('moments_hu6', prop.moments_hu[5]),
                                 ('moments_hu7', prop.moments_hu[6]),

                                 ('euler_number', prop.euler_number),  # miscellaneous

                                 ('countCoords', len(prop.coords))))  # eventually grab these coordinates?

    return max_area_dict


if __name__ == '__main__':
    # imagePropertiesList = []
    BASE_DIR = '/plankton'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    TEST_DIR = os.path.join(BASE_DIR, 'test')
    directory_names = os.listdir(TRAIN_DIR)

    csv_name = 'train_img_props.csv'

    with open(csv_name, 'w') as f:
        needs_header = True
        for train_index in range(len(directory_names)):
            folder = directory_names[train_index]
            basedir = os.path.join(TRAIN_DIR, folder)
            filenames = os.listdir(basedir)

            print(train_index, folder, len(filenames))
            for index in range(len(filenames)):
                filename = filenames[index]
                fullname = os.path.join(basedir, filename)

                image_property_dict = get_max_area_dict(fullname, folder)

                if needs_header:
                    needs_header = False
                    print('\t'.join(image_property_dict.iterkeys()), file=f)
                print('\t'.join(str(v) for v in image_property_dict.itervalues()), file=f)

                # imagePropertiesList.append(image_property_dict)

    # csv_name = 'test_img_props.csv'
    #
    # test_file_names = os.listdir(TEST_DIR)
    #
    # for test_index in range(len(test_file_names)):
    #     filename = test_file_names[test_index]
    #     if test_index % 1000 == 0:
    #         print(test_index, '/', len(test_file_names))
    #
    #     fullname = os.path.join(TEST_DIR, filename)
    #
    #     image_property_dict = get_max_area_dict(fullname)
    #
    #     imagePropertiesList.append(image_property_dict)

    # df = pandas.DataFrame(imagePropertiesList)
    # df.to_csv(csv_name, sep='\t', index=False)