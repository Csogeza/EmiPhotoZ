from astropy.cosmology import FlatLambdaCDM
import scipy.interpolate
import scipy.signal
import os, os.path
import numpy

LIGHTSPEED = 299792.458  # km s-1
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

# Luminosity function for drawing realistic absolute magnitudes
def schechter_func(M, phi, m0, alp):
    return (
        0.4
        * numpy.log(10)
        * phi
        * (numpy.power(10, -0.4 * (M - m0) * (alp + 1)))
        * numpy.exp(-(numpy.power(10, -0.4 * (M - m0))))
    )

	
# Function that finds the closest wavelength gridpoint to a given value
def find_closest_wl_index(wl, measure_points):
    diff = []
    for i in range(len(measure_points)):
        diff.append(abs(measure_points[i] - wl))
    val, idx = min((val, idx) for (idx, val) in enumerate(diff))
    return idx

	
# Resample spectra for the new_wl grid and find the corresponding wavelength indices
def resample_spec_indfind(orig_wl, orig_fl, new_wl):
    low = min(orig_wl)
    temp_il = find_closest_wl_index(low, new_wl)
    low_ind = temp_il + 1
    up = max(orig_wl)
    temp_iu = find_closest_wl_index(up, new_wl)
    up_ind = temp_iu - 1
    new_fl_line = scipy.interpolate.interp1d(orig_wl, orig_fl)(new_wl[low_ind:up_ind])
    temp_fl = numpy.zeros((len(new_wl)))
    temp_fl[low_ind:up_ind] = new_fl_line[:]
    return temp_fl, low_ind, up_ind

	
# Resample routine for emission lines
def resample_spec_line(orig_wl, orig_fl, new_wl, low_ind, up_ind):

    new_fl_line = scipy.interpolate.interp1d(orig_wl, orig_fl)(new_wl[low_ind:up_ind])
    temp_fl = numpy.zeros((len(new_wl)))
    temp_fl[low_ind:up_ind] = new_fl_line[:]
    return temp_fl

	
# Load the BC03 templates from a given folder and order them into arrays
def load_templates_1(Z, age):
    tmp = numpy.loadtxt("./templates/1/m42.spec", comments="#")

    temp_wl = tmp[:, 0]
    temp_fl = numpy.zeros((len(Z), len(age), tmp.shape[0]))

    temp_fl[0, :, :] = numpy.loadtxt(
        "./templates/3/m42.spec", comments="#"
    ).transpose()[1:, :]
    temp_fl[1, :, :] = numpy.loadtxt(
        "./templates/3/m52.spec", comments="#"
    ).transpose()[1:, :]
    temp_fl[2, :, :] = numpy.loadtxt(
        "./templates/3/m62.spec", comments="#"
    ).transpose()[1:, :]
    temp_fl[3, :, :] = numpy.loadtxt(
        "./templates/3/m72.spec", comments="#"
    ).transpose()[1:, :]

    return temp_wl, temp_fl

	
# Apply extinction on the spectra
def extinction_bc03(tau_V, mu, wl):
    wltr = numpy.power(wl / 5500.0, -0.7)
    fm = numpy.exp(-mu * tau_V * wltr)
    f1 = numpy.exp(-tau_V * wltr)
    return fm, f1

	
# Simple resample routine
def resample_spec(orig_wl, orig_fl, new_wl):
    new_fl = scipy.interpolate.interp1d(orig_wl, orig_fl)(new_wl)
    return new_fl

	
# Applying logarithmic binning on the spectra
def spec_log_binning(temp_wl, temp_fl):
    temp_log_wl = numpy.logspace(3, 4.2, 11000)
    temp_log_fl = numpy.zeros(
        (temp_fl.shape[0], temp_fl.shape[1], temp_log_wl.shape[0])
    )

    for z in range(0, temp_fl.shape[0]):
        for a in range(0, temp_fl.shape[1]):
            temp_log_fl[z, a, :] = resample_spec(temp_wl, temp_fl[z, a, :], temp_log_wl)

    return temp_log_wl, temp_log_fl

	
# Calculating velocity dispersion kernel
def create_vdisp_kernel(log_wl, vdisp=200, N=21):
    # velocity dispersion per bin
    vv = LIGHTSPEED * (log_wl[1] / log_wl[0] - 1)
    sigma = vdisp / vv
    k = scipy.signal.gaussian(N, sigma)
    return k / numpy.sum(k)


# Convolving velocity dispersion kernels
def convolve_vdisp_kernel(log_wl, log_fl, vdisp=200, N=21):
    k = create_vdisp_kernel(log_wl, vdisp, N)
    cc = numpy.zeros(log_fl.shape)
    for z in range(0, log_fl.shape[0]):
        for a in range(0, log_fl.shape[1]):
            cc[z, a, :] = numpy.convolve(log_fl[z, a, :], k, mode="same")
    return cc


# Model evaluation routine for a given set of parameters; apply convolution with the
# right velocity dispersion kernel, then reapply extinction, then add resulting flux series
def eval_model(temp_log_wl, temp_log_fl, z, coeff, vdisp, tau_V, mu):
    conv_log_fl = convolve_vdisp_kernel(temp_log_wl, temp_log_fl, vdisp)
    fm, f1 = extinction_bc03(tau_V, mu, temp_log_wl)
    model_log_fl = f1 * numpy.dot(conv_log_fl[z, 0, :], coeff[0]) + fm * numpy.dot(
        conv_log_fl[z, 1:, :].transpose(), coeff[1:]
    )
    return model_log_fl

	
# Fitting for the coefficients of the SSP-s using NNLS
def fit_nnls_nm(spec_wl, spec_fl, temp_log_wl, temp_log_fl, z, vdisp, tau_V, mu):

    # convolve and resample to grid of spec
    conv_log_fl = convolve_vdisp_kernel(temp_log_wl, temp_log_fl, vdisp)
    conv_fl = resample_spec(temp_log_wl, conv_log_fl, spec_wl)

    # apply extinction
    fm, f1 = extinction_bc03(tau_V, mu, spec_wl)
    conv_fl[z, 0, :] = conv_fl[z, 0, :] * f1
    conv_fl[z, 1:, :] = conv_fl[z, 1:, :] * fm

    A = conv_fl[z].T
    b = spec_fl

    coeff, chi2 = scipy.optimize.nnls(A, b)

    cont_fl = numpy.dot(conv_fl[z, :, :].transpose(), coeff)

    return coeff, chi2, cont_fl

	
# Refitting the extinction parameters based on a previous NNLS fitting, using
# Nelder-Mead method
def fit_all_nm(
    spec_wl, spec_fl, temp_log_wl, temp_log_fl, z, vdisp=220, tau_V=0.8, mu=0.3
):
    def cost(x):
        coeff, chi2, cont_fl = fit_nnls_nm(
            spec_wl, spec_fl, temp_log_wl, temp_log_fl, z, vdisp, x[0], mu
        )
        return chi2

    x = [tau_V]
    res = scipy.optimize.minimize(
        cost, x, method="Nelder-Mead", options={"maxiter": 100}
    )
    [tau_V] = res.x

    coeff, chi2, cont_fl = fit_nnls_nm(
        spec_wl, spec_fl, temp_log_wl, temp_log_fl, z, vdisp, tau_V, mu
    )

    return coeff, res.x, chi2

	
# Calculating the comoving volume differences at given redshifts
def dV_c(z):
    E = cosmo.efunc(z)
    D_C = cosmo.comoving_distance(z)
    dvc = cosmo.hubble_distance * 4 * numpy.pi * D_C ** 2 * E
    return dvc

	
# Drawing redshifts with uniform distribution according to comoving redshift 
def gen_redshift(n, z1, z2):
    gen_z = numpy.zeros((n))
    gen_dl = numpy.zeros((n))
    pzmax = dV_c(z2)
    for i in range(n):
        y = 1
        p = 0
        while y > p:
            rz = z1 + (z2 - z1) * numpy.random.uniform(0, 1)
            p = dV_c(rz)
            y = pzmax * numpy.random.uniform(0, 1)
        gen_z[i] = rz
        gen_dl[i] = cosmo.luminosity_distance(rz).value

    return gen_z


# Drawing absolute magnitudes from the Schechter function
def gen_M_sample(M_min, M_max, num, M_star, a):
    phi_max = 1 / schechter_func(M_max, 1, M_star, a)
    i = 0
    magnitudes = numpy.zeros((num))
    while i < num:
        rand_magni = numpy.random.uniform(M_min, M_max)
        Value = numpy.random.random()

        if Value < schechter_func(rand_magni, phi_max, M_star, a):
            magnitudes[i] = rand_magni
            i += 1
    return magnitudes

	
# Pairing absolute magnitude - redshift values and keeping the ones that
# meet the flux limit criteria
def gal_sample(n, cosmo, zz, MM, mLim):
    z = numpy.zeros(n)
    M = numpy.zeros(n)
    m = numpy.zeros(n)
    for i in range(0, n):
        while True:
            zR = numpy.random.choice(zz)
            MR = numpy.random.choice(MM)
            DL = cosmo.luminosity_distance(zR).value
            DM = 5 * numpy.log10(DL)
            mR = MR + DM + 25
            if mR < mLim:
                z[i] = zR
                M[i] = MR
                m[i] = mR
                break
    return z, M, m

	
# Drawing continuum PC coefficients from the GMM-s
def generate_pca_coeffs(M, gmm_dict):
    if M < -23.5:
        PComponents = gmm_dict[0].sample(1)[0][0]
    elif M > -17:
        PComponents = gmm_dict[len(gmm_dict) - 1].sample(1)[0][0]
    else:
        for i in range(13):
            if M < -23.0 + i / 2:
                PComponents = gmm_dict[1 + i].sample(1)[0][0]
                break
    return PComponents

	
# Sampling the emission line GMM-s
def generate_line_ews(ind, emiss_gmm):
    EW = emiss_gmm[ind].sample(1)[0][0]
    return EW

	
# Generating continuum PC coefficients, then extending the simulated continuum towards UV 
# and IR based on BC03 models
def generate_continuum(M, avg, eig5, gmm_dict, temp_log_wl, temp_log_fl):
    spectrum = numpy.zeros((eig5.shape[1], 2))

    PComponents = generate_pca_coeffs(M, gmm_dict)

    spectrum[:, 0] = avg[:, 0]
    spectrum[:, 1] = avg[:, 1] + numpy.dot(eig5.T, PComponents)

    chim = numpy.zeros((4))
    taus = numpy.zeros((4))

    for i in range(4):
        lab_Z = i
        coeff, taus[i], chim[i] = fit_all_nm(
            spectrum[:, 0],
            spectrum[:, 1],
            temp_log_wl,
            temp_log_fl,
            lab_Z,
            220,
            0.8,
            0.3,
        )
    val, idx = min((val, idx) for (idx, val) in enumerate(chim))

    coeff, chi2, tem = fit_nnls_nm(
        spectrum[:, 0],
        spectrum[:, 1],
        temp_log_wl,
        temp_log_fl,
        idx,
        220,
        taus[idx],
        0.3,
    )
    final_model = eval_model(temp_log_wl, temp_log_fl, idx, coeff, 220, taus[idx], 0.3)

    return final_model, PComponents, spectrum


# Simple routine for the Gaussian profile
def gauss(x, sig, l0):
    return (
        1
        / (numpy.sqrt(2 * numpy.pi) * sig)
        * numpy.exp(-((x - l0) ** 2) / (2 * sig ** 2))
    )


# Rountine for fitting an exponential function with a cutoff at a given value
def fit_exp(x, a, b, c, d):
    try:
        val = list(a * numpy.exp(-b * x + c) + d)
        for i in range(len(val)):
            if val[i] < 0:
                val[i] = 0
    except TypeError:
        val = a * numpy.exp(-b * x + c) + d
        if val < 0:
            val = 0
    return val


# Finding the closest continuum kMeans cluster to a continuum
def find_nearest_cluster(PComponents, centr):

    distances = {}
    for l in range(0, centr.shape[1]):
        distances[l] = 0
        temp = 0
        for m in range(len(PComponents)):
            temp = temp + (PComponents[m] - centr[m, l]) ** 2
        distances[l] = numpy.sqrt(temp)
        del temp
    minimal = 30
    for l in range(0, centr.shape[1]):
        temp = distances[l]
        if temp < minimal:
            minimal = temp
            I = l
        del temp
    ind = I
    return ind


# Adding emission lines to strong emission line galaxies based on the line PCA
def add_lines(
    ind,
    sig_lines,
    low_indices,
    up_indices,
    l_down,
    l_up,
    V_emi,
    line_avg,
    emi_wl,
    spec_bin,
    spectrum,
    emi_gmm,
    strength_coeff=1,
):
    Line_PComp = emi_gmm[ind].sample(1)[0][0]
    logEWs = numpy.dot(V_emi.T, Line_PComp) + line_avg[:, 1]
    EW = numpy.exp(logEWs)

    amplitudes = numpy.zeros((10))
    idx = numpy.zeros((10))
    cont_flux = numpy.zeros((10))

    for i in range(10):
        idx[i] = find_closest_wl_index(emi_wl[i], spec_bin)
        cont_flux[i] = numpy.mean(spectrum[int(idx[i] - 4) : int(idx[i] + 4)])

    res_fl = {}
    for i in range(0, 10):
        lamb = numpy.linspace(l_down[i], l_up[i], 1000)
        Flamb = numpy.zeros((1000))
        for j in range(1000):
            amplitudes[i] = cont_flux[i] * EW[i]
            Flamb[j] = (
                amplitudes[i] * strength_coeff * gauss(lamb[j], sig_lines[i], emi_wl[i])
            )
        res_fl[i] = resample_spec_line(
            lamb, Flamb, spec_bin, int(low_indices[i]), int(up_indices[i])
        )
    line_flux = numpy.zeros((len(spec_bin)))
    for i in range(10):
        line_flux = line_flux + res_fl[i]

    return line_flux, logEWs


# Checking the emission probability maps; selecting the correct map based on the absolute
# magnitude, then finding the probability of emission lines in the bin corresponding to the
# continuum PCs
def get_emission_weight(
    PComponents, M, wms, M_v=[-24, -16.5], x_pc=[-40, 30], y_pc=[-8, 8], stp=10
):

    x_seps = numpy.linspace(x_pc[0], x_pc[1], stp)
    y_seps = numpy.linspace(y_pc[0], y_pc[1], stp)
    M_seps = numpy.linspace(M_v[0], M_v[1], len(wms) + 1)
    x_values = abs(
        numpy.linspace(
            (x_seps[0] + x_seps[1]) / 2, (x_seps[-2] + x_seps[-1]) / 2, stp - 1
        )
        - PComponents[0]
    )
    y_values = abs(
        numpy.linspace(
            (y_seps[0] + y_seps[1]) / 2, (y_seps[-2] + y_seps[-1]) / 2, stp - 1
        )
        - PComponents[1]
    )
    M_values = abs(
        numpy.linspace(
            (M_seps[0] + M_seps[1]) / 2, (M_seps[-2] + M_seps[-1]) / 2, len(wms)
        )
        - M
    )

    min_ind_x, min_val_x = min(enumerate(x_values), key=lambda p: p[1])
    min_ind_y, min_val_y = min(enumerate(y_values), key=lambda p: p[1])
    min_ind_M, min_val_M = min(enumerate(M_values), key=lambda p: p[1])

    return wms[min_ind_M][min_ind_x, min_ind_y]


# If the galaxy is not strong emission line galaxy; add only Halpha: check the continuum
# PC distance from the reference point, then infer a mean EW based on the exponential fit,
# then add a random term
def add_hI_only(
    cont_pc,
    sig_lines,
    low_indices,
    up_indices,
    l_down,
    l_up,
    spec_bin,
    spectrum,
    popt,
    emi_wl,
):

    x_coord = (cont_pc[0] + 40) / 70 * 30   # the hardcoded values represent the reference position
    y_coord = (cont_pc[1] + 8) / 16 * 30    # (put the reference point to (0,0), then scale the lengths)
    distance = numpy.sqrt(x_coord ** 2 + y_coord ** 2)

    if distance > 15 and distance < 28.5:   # at different PC "distances" the random scatter had different magnitudes
        temp = fit_exp(distance, *popt) + numpy.random.normal(
            0, 0.7 * (28.5 - distance)
        )
        if numpy.random.uniform(0, 1) < 13136 / 42977 or temp < 0:  # some galaxies did not even have emission lines
            temp = 0												# regardless of distance; their fraction is set
    elif distance > 28.5:											# empirically
        temp = 0     # Large "distance" = LRG ==> no line
    else:
        temp = fit_exp(distance, *popt) + (
            numpy.random.normal(0, 0.7 * (28.5 - 15) + 10 * (distance - 15) ** 2)
        )			# We had to scale the scatter for the closer regions
        if temp < 0:
            temp = 0

    EW = temp
	# Fint the closest wavelength bin; add a Gaussian centered on it
    idx = find_closest_wl_index(emi_wl[6], spec_bin)
    cont_f = numpy.mean(spectrum[int(idx - 4) : int(idx + 4)])

    res_fl = {}
    lamb = numpy.linspace(l_down[6], l_up[6], 1000)
    Flamb = numpy.zeros((1000))
    for j in range(1000):
        amplitude = cont_f * EW
        Flamb[j] = amplitude * gauss(lamb[j], sig_lines[6], emi_wl[6])

    res_fl = resample_spec_line(
        lamb, Flamb, spec_bin, int(low_indices[6]), int(up_indices[6])
    )
    line_flux = numpy.zeros((len(spec_bin)))
    line_flux = line_flux + res_fl

    return line_flux, EW

	
# The whole procedure; supply every variable, and it generates a given amount of spectra
def processInput(
    run_ind,
    result_spectrum,
    galaxies,
    result_dir,
    popt,
    cont_avg,
    cont_eig5,
    gmm,
    emi_gmm,
    new_spec_wl,
    temp_log_wl,
    temp_log_fl,
    weight_maps,
    centr,
    sig_lines,
    low_indices,
    up_indices,
    l_down,
    l_up,
    V_emi,
    atlag,
    emi_wl,
):

    new_wl_uv = new_spec_wl[:4541]    # Hardcoded values; only work for the galaxy set we used
    new_wl_inf = new_spec_wl[-7567:]  # Basically tell the program, from where will we need to stich the
    numpy.random.seed()				  # simulated spectra with the BC03 models; these are the indices of
    M = galaxies[run_ind, 1]		  # lower and upper end of the wavelength grid of the PCA eigenspectra
    while True:						  # in the wavelength grid we defined here
        try:
            model, PComponents, spectrum = generate_continuum(
                M, cont_avg, cont_eig5, gmm, temp_log_wl, temp_log_fl
            )
            break
        except RuntimeError:  # sometimes it can fail at Nelder-Mead; simulate completely new spectra
            continue
	# Stitch the PCA spectra with the BC03 models
    res_mod_uv = resample_spec(temp_log_wl, model, new_wl_uv)
    res_mod_inf = resample_spec(temp_log_wl, model, new_wl_inf)
    new_spec_fl_t = numpy.append(
        res_mod_uv - res_mod_uv[-1] + spectrum[0, 1], spectrum[1:, 1]
    )
    new_spec_fl = numpy.append(
        new_spec_fl_t - new_spec_fl_t[-1] + res_mod_inf[0], res_mod_inf[1:]
    )
	# Determine whether the simulated galaxy is of strong emission line type
    if numpy.random.uniform(0, 1) < get_emission_weight(PComponents, M, weight_maps):

        temp_type = "Emission"

    else:

        temp_type = None

    if temp_type == "Emission":

        ind = find_nearest_cluster(PComponents, centr)

        line_flux, logEWs = add_lines(
            ind,
            sig_lines,
            low_indices,
            up_indices,
            l_down,
            l_up,
            V_emi,
            atlag,
            emi_wl,
            new_spec_wl,
            new_spec_fl,
            emi_gmm,
        )

    else:

        line_flux, logEWs = add_hI_only(
            PComponents,
            sig_lines,
            low_indices,
            up_indices,
            l_down,
            l_up,
            new_spec_wl,
            new_spec_fl,
            popt,
            emi_wl,
        )

    spec_bin = new_spec_wl
    Specz = numpy.empty((len(spec_bin), 2))

    Specz[:, 0] = (1.0 + galaxies[run_ind, 0]) * spec_bin[:]
    Specz[:, 1] = line_flux[:] + new_spec_fl[:]
    result_spectrum[run_ind] = {
        "M": galaxies[run_ind, 1],
        "z": galaxies[run_ind, 0],
        "Spectrum": Specz,
        "Continuum": new_spec_fl,
        "Lines": line_flux,
        "PCs": PComponents,
        "LineLogEW": logEWs,
    }
	# Dump the results in a separate .npy file
    numpy.save(result_dir + str(run_ind) + ".npy", result_spectrum)

    pass

# Find the edges of the line in the wavelength grid
def find_line_boundary_indices(spec_bin, emi_wl, sigma, l_up, l_down):

    low_indices = numpy.zeros((emi_wl.shape[0]))
    up_indices = low_indices.copy()

    res_fl = {}

    for i in range(emi_wl.shape[0]):

        lamb = numpy.linspace(l_down[i], l_up[i], 1000)
        Flamb = numpy.zeros((1000))
        res_fl[i], low_indices[i], up_indices[i] = resample_spec_indfind(
            lamb, Flamb, spec_bin
        )

    return res_fl, low_indices, up_indices
