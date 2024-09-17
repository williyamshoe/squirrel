"""This module contains class and functions for general use."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pafit.fit_kinematic_pa import fit_kinematic_pa


def register_sauron_colormap():
    """
    Regitsr the 'sauron' and 'sauron_r' colormaps in Matplotlib

    """
    cdict = {'red':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.4,    0.4),
                 (0.414,   0.5,    0.5),
                 (0.463,   0.3,    0.3),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.9,    0.9)],
        'green':[(0.000,   0.01,   0.01),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)],
         'blue':[(0.000,   0.01,   0.01),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.0,    0.0),
                 (0.590,   0.0,    0.0),
                 (0.668,   0.0,    0.0),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.9,    0.9)]
         }

    rdict = {'red':[(0.000,   0.9,    0.9),
                 (0.170,   1.0,    1.0),
                 (0.336,   1.0,    1.0),
                 (0.414,   1.0,    1.0),
                 (0.463,   0.7,    0.7),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.3,    0.3),
                 (0.590,   0.5,    0.5),
                 (0.668,   0.4,    0.4),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
        'green':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.85,   0.85),
                 (0.414,   1.0,    1.0),
                 (0.463,   1.0,    1.0),
                 (0.502,   0.9,    0.9),
                 (0.541,   1.0,    1.0),
                 (0.590,   1.0,    1.0),
                 (0.668,   0.85,   0.85),
                 (0.834,   0.0,    0.0),
                 (1.000,   0.01,   0.01)],
         'blue':[(0.000,   0.9,    0.9),
                 (0.170,   0.0,    0.0),
                 (0.336,   0.0,    0.0),
                 (0.414,   0.0,    0.0),
                 (0.463,   0.0,    0.0),
                 (0.502,   0.0,    0.0),
                 (0.541,   0.7,    0.7),
                 (0.590,   1.0,    1.0),
                 (0.668,   1.0,    1.0),
                 (0.834,   1.0,    1.0),
                 (1.000,   0.01,   0.01)]
         }

    sauron = colors.LinearSegmentedColormap('sauron', cdict)
    sauron_r = colors.LinearSegmentedColormap('sauron_r', rdict)
    plt.register_cmap(cmap=sauron)
    plt.register_cmap(cmap=sauron_r)

register_sauron_colormap()

def plot_kinematic_maps(voronoi_binning_output, bin_centers, bin_kinematics, radius_in_pixels, pixel_scale, reff,
                        show_bin_num=True, vminmax=np.zeros(4), mask_error=20, annular_global_templates=True):
    '''
    Function to make 2D maps of kinematic components and errors of all bins measured in "ppxf_bin_spectra" function.
    '''
    annular_radii=np.array([0.5, 1., 1.5]) * reff * pixel_scale

    # make arrays of kinematic components and error of size number of pixels
    VD_array = np.zeros(voronoi_binning_output.shape[0])
    dVD_array = np.zeros(voronoi_binning_output.shape[0])
    V_array = np.zeros(voronoi_binning_output.shape[0])
    dV_array = np.zeros(voronoi_binning_output.shape[0])

    # loop through each pixel and assign the kinematics values from the bin measurements
    for i in range(voronoi_binning_output.shape[0]):
        # num is bin number
        num = int(voronoi_binning_output.T[2][i])
        # take bin kinematics
        vd = bin_kinematics[num][1]
        dvd = bin_kinematics[num][3]
        v = bin_kinematics[num][0]
        dv = bin_kinematics[num][2]
        # update the array with the pixel's assigned kinematics
        VD_array[i] = vd
        dVD_array[i] = dvd
        V_array[i] = v
        dV_array[i] = dv

    # stack the pixel kinematics with the pixel bin information
    pixel_details = np.vstack((voronoi_binning_output.T, VD_array, dVD_array, V_array, dV_array))

    # dimension of the square cropped datacube
    dim = radius_in_pixels * 2 + 1

    # create the 2D kinematic maps by looping through each pixel and taking teh values from "pixel_details", pixels with no kinematics are 'nan'
    # velocity dispersion
    VD_2d = np.zeros((dim, dim))
    VD_2d[:] = np.nan
    for i in range(pixel_details.shape[1]):
        VD_2d[int(pixel_details[1][i])][int(pixel_details[0][i])] = pixel_details[3][i]

    # error in velocity dispersion
    dVD_2d = np.zeros((dim, dim))
    dVD_2d[:] = np.nan
    for i in range(pixel_details.shape[1]):
        dVD_2d[int(pixel_details[1][i])][int(pixel_details[0][i])] = pixel_details[4][i]

    # velocity
    V_2d = np.zeros((dim, dim))
    V_2d[:] = np.nan
    for i in range(pixel_details.shape[1]):
        V_2d[int(pixel_details[1][i])][int(pixel_details[0][i])] = pixel_details[5][i]

    # error in velocity
    dV_2d = np.zeros((dim, dim))
    dV_2d[:] = np.nan
    for i in range(pixel_details.shape[1]):
        dV_2d[int(pixel_details[1][i])][int(pixel_details[0][i])] = pixel_details[6][i]

    good_vd = dVD_2d < mask_error
    good_v = dV_2d < mask_error
    good_bins = good_vd * good_v
    VD = np.copy(VD_2d)
    VD[~good_bins] = np.nan

    # extent
    extent = np.array([-1, 1, -1, 1]) * \
             (VD_2d.shape[0] // 2) * pixel_scale  # will give half of image size rounded down (e.g. 43 -> 21)
    # should make it ~6x6 arcsec if cropped to 3 arcsec radius

    if show_bin_num:
        plt.figure();

        extent, p = display_pixels(voronoi_binning_output[:, 0], voronoi_binning_output[:, 1],
                                   [bin_kinematics[int(ii), 0] for ii in voronoi_binning_output[:, 2]], pixel_scale)

        # p = plt.imshow(VD_2d, origin='lower', cmap='gist_rainbow', extent=extent)
        if annular_global_templates:
            # plot circular annuli
            for i, radius in enumerate(annular_radii):
                circle = plt.Circle((0, 0), radius, color='k', fill=False, linestyle='--')
                ax = plt.gca()
                ax.add_patch(circle)
        # plt.imshow(~good_bins, origin='lower', alpha=0.7, cmap='Greys', extent=extent)
        cbar1 = plt.colorbar(p)
        cbar1.set_label(r'$\sigma$ [km/s]')
        for i, row in enumerate(bin_centers):
            # bin_centers_ = (row - (VD_2d.shape[0] // 2)) * pixel_scale
            plt.annotate(int(i), row, fontsize=10, annotation_clip=False, ha='center')

        # plt.savefig(target_dir + obj_name + '_VD.png')
        plt.pause(1)
        plt.clf()

    # if any(vminmax > 0):
    #     v_min = vminmax[0]
    #     v_max = vminmax[1]
    #     vd_min = vminmax[2]
    #     vd_max = vminmax[3]
    # else:
    #     v_min = -100
    #     v_max = 100
    #     vd_min = np.nanmin(VD) - 5
    #     vd_max = np.nanmax(VD) + 5
    #
    # # velocity dispersion
    # plt.figure();
    # plt.imshow(VD, origin='lower', cmap='sauron', vmin=vd_min, vmax=vd_max, extent=extent);
    # if annular_global_templates:
    #     # plot circular annuli
    #     for i, radius in enumerate(annular_radii):
    #         circle = plt.Circle((0, 0), radius, color='k', fill=False, linestyle='--')
    #         ax = plt.gca()
    #         ax.add_patch(circle)
    #         # plt.imshow(bad_bins, origin='lower', cmap='Greys')
    # cbar1 = plt.colorbar()
    # cbar1.set_label(r'$\sigma$ [km/s]')
    # # plt.savefig(target_dir + obj_name + '_VD.png')
    # plt.pause(1)
    # plt.clf()
    #
    # # error in velocity dispersion
    # plt.figure();
    # dVD = np.copy(dVD_2d)
    # dVD[~good_bins] = np.nan
    # plt.imshow(dVD, origin='lower', cmap='sauron', vmin=0, vmax=mask_error, extent=extent);
    # if annular_global_templates:
    #     # plot circular annuli
    #     for i, radius in enumerate(annular_radii):
    #         circle = plt.Circle((0, 0), radius, color='k', fill=False, linestyle='--')
    #         ax = plt.gca()
    #         ax.add_patch(circle)
    # cbar2 = plt.colorbar()
    # cbar2.set_label(r'd$\sigma$ [km/s]')
    # # plt.savefig(target_dir + obj_name + '_dVD.png')
    # plt.pause(1)
    # plt.clf()
    #
    # # subtract the "bulk" velocity, small offset in galaxy velocity from redshift error
    # bulk = np.nanmedian(V_2d)
    # # calculate a better bulk velocity
    # pa, pa_err, vsyst = fit_kinematic_pa(bin_centers[:, 0] - 21, bin_centers[:, 1] - 21,
    #                                      bin_kinematics[:, 0] - bulk)
    # bulk += vsyst
    #
    # # mean velocity
    # plt.figure();
    # V = np.copy(V_2d)
    # V[~good_bins] = np.nan
    # plt.imshow(V - bulk, origin='lower', cmap='sauron', vmin=v_min, vmax=v_max, extent=extent);
    # if annular_global_templates:
    #     # plot circular annuli
    #     for i, radius in enumerate(annular_radii):
    #         circle = plt.Circle((0, 0), radius, color='k', fill=False, linestyle='--')
    #         ax = plt.gca()
    #         ax.add_patch(circle)
    # cbar3 = plt.colorbar()
    # cbar3.set_label(r'Vel [km/s]')
    # plt.title("Velocity map")
    # # plt.savefig(target_dir + obj_name + '_V.png')
    # plt.pause(1)
    # plt.clf()
    #
    # # error in velocity
    # plt.figure();
    # dV = np.copy(dV_2d)
    # dV[~good_bins] = np.nan
    # plt.imshow(dV, origin='lower', cmap='sauron', vmin=0, vmax=mask_error, extent=extent);
    # if annular_global_templates:
    #     # plot circular annuli
    #     for i, radius in enumerate(annular_radii):
    #         circle = plt.Circle((0, 0), radius, color='k', fill=False, linestyle='--')
    #         ax = plt.gca()
    #         ax.add_patch(circle)
    # cbar4 = plt.colorbar()
    # cbar4.set_label(r'dVel [km/s]')
    # plt.title("error on velocity")
    # plt.pause(1)
    # plt.clf()

def display_pixels(x, y, counts, pixelsize):
    """
    Display pixels at coordinates (x, y) coloured with "counts".
    This routine is fast but not fully general as it assumes the spaxels
    are on a regular grid. This needs not be the case for Voronoi binning.

    """
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin)/pixelsize) + 1)
    ny = int(round((ymax - ymin)/pixelsize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin)/pixelsize).astype(int)
    k = np.round((y - ymin)/pixelsize).astype(int)
    img[j, k] = counts

    extent = [xmin - pixelsize / 2, xmax + pixelsize / 2,
              ymin - pixelsize / 2, ymax + pixelsize / 2]

    return (extent, plt.imshow(np.rot90(img), interpolation='nearest', cmap='sauron',
                      extent=extent))
