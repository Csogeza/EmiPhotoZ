import numpy
import scipy.interpolate
import matplotlib.pyplot


# Calculate the emission line log EW average for the sample
def average_log_emission(emiss_logEW_array, emission_wl):

    avg_emiss = numpy.empty((len(emission_w), 2))
    avg_emiss[:, 0] = array(emission_wl)[:]
    avg_emiss[:, 1] = numpy.mean(emiss_logEW_array, axis=0)

    return avg_emiss


# Calculate the average spectrum
def average_spec(normed_spec):

    avg_spec = numpy.zeros(normed_spec[0].shape)

    avg_spec[:, 0] = normed_spec[0][:, 0]

    for j in range(len(normed_spec)):

        avg_spec[:, 1] = avg_spec[:, 1] + normed_spec[j][:, 1]

    avg_spec[:, 1] = avg_spec[:, 1] / (j + 1)

    return avg_spec


# Subtract the average of the individual logEW vectors
def get_reduced_logEW(emiss_logEW_array, avg_emiss):

    red_logEW_array = numpy.zeros(emiss_logEW_array.shape)

    for i in range(emiss_logEW_array.shape[0]):

        red_logEW_array[i] = emiss_logEW_array[i] - avg_emiss[:, 1]

    return red_logEW_array


# Subtract the average of the individual continuum spectra
def get_reduced_spectra(normed_spec, avg_spec):

    red_spec = {}

    for j in range(len(normed_spec)):

        temp = normed_spec[j]

        temp[:, 1] = temp[:, 1] - avg_spec[:, 1]

        red_spec[j] = temp

        del temp

    return red_spec


# Get emission line logarithms
def logaritmize_EW(emiss_lines):

    emiss_logEW_array = numpy.zeros((len(emiss_lines), 10))

    for i in range(len(emiss_lines)):

        emiss_logEW_array[i] = numpy.log(emiss_lines[i])

    return emiss_logEW_array


# Use the normalize_spec on the set of models
def normalize_model(spectra, new_wl):
    normed_spec = {}
    k = 0
    for j in range(len(spectra)):
        if spectra[j]["Model"] is not None and max(spectra[j]["Model"][:]) > 0.01:

            temp = normalize_spec(
                numpy.vstack((spectra[j]["Spectrum"][:, 0], spectra[j]["Model"][:])).T
            )
            temp_fl = resample_spec(temp[:, 0], temp[:, 1], new_wl)
            normed_spec[k] = numpy.vstack((new_wl, temp_fl)).T
            k += 1
        else:
            continue

    return normed_spec


# Normalise a spectrum by setting the average of medians of four featureless wavelength region to 1
def normalize_spec(temp):

    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    n = 0
    for k in temp[:, 0]:
        if k > 4250 and k < 4300:
            if m1 == 0:
                norm_region1 = temp[n, 1]
                m1 += 1
            else:
                norm_region1 = numpy.append(norm_region1, temp[n, 1])

        if k > 4600 and k < 4800:
            if m2 == 0:
                norm_region2 = temp[n, 1]
                m2 += 1
            else:
                norm_region2 = numpy.append(norm_region2, temp[n, 1])

        if k > 5400 and k < 5500:
            if m3 == 0:
                norm_region3 = temp[n, 1]
                m3 += 1
            else:
                norm_region3 = numpy.append(norm_region3, temp[n, 1])

        if k > 5600 and k < 5800:
            if m4 == 0:
                norm_region4 = temp[n, 1]
                m4 += 1
            else:
                norm_region4 = numpy.append(norm_region4, temp[n, 1])

        n += 1

    med1 = numpy.median(norm_region1)
    med2 = numpy.median(norm_region2)
    med3 = numpy.median(norm_region3)
    med4 = numpy.median(norm_region4)
    avgmed = (med1 + med2 + med3 + med4) / 4

    temp[:, 1] = temp[:, 1] / avgmed
    return temp


# Simple resample routine
def resample_spec(orig_wl, orig_fl, new_wl):

    new_fl = scipy.interpolate.interp1d(orig_wl, orig_fl)(new_wl)

    return new_fl


# Run SVD on the residual line datavectors
def run_line_pca(red_logEW_array, numEig):

    xxT = numpy.dot(red_logEW_array.T, red_logEW_array)

    U, s, V = numpy.linalg.svd(xxT)

    E_PCs = numpy.dot(V[0:numEig, :], red_logEW_array.T)

    return V, E_PCs


# Run SVD on the residual continuum datavectors
def run_pca(red_spec, new_wl, numEigV):

    correlation_mtx = numpy.zeros((len(new_wl), len(red_spec)))

    for j in range(len(red_spec)):

        temp = red_spec[j]

        correlation_mtx[:, j] = temp[:, 1]

    xxT = numpy.matmul(correlation_mtx, correlation_mtx.T)

    U_p, s_p, V_p = numpy.linalg.svd(xxT)

    eig5 = V_p[0:numEigV, :]

    PCs = numpy.dot(eig5, correlation_mtx)

    return eig5, PCs


# Separate the strong emission line galaxies from the ones that do not exhibit every emission line
def separate_emission_galaxies(spectra, line_data, PCs):

    emiss_lines = {}
    emiss_pcs = []
    no_lines = {}
    no_pcs = []

    k = 0
    n = 0
    j = 0

    for i in range(len(line_data)):

        if spectra[i]["Model"] is not None and max(spectra[i]["Model"]) > 0.01:

            if min(line_data[i]) > 0:

                emiss_lines[k] = line_data[i]

                emiss_pcs.append(PCs.T[j])

                k += 1
                j += 1

            else:

                no_lines[n] = line_data[i]

                no_pcs.append(PCs.T[j])

                n += 1
                j += 1

    return emiss_lines, emiss_pcs, no_lines, no_pcs
