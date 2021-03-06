import numpy as np

####################################################################################################

RandState = np.random.RandomState(67)

####################################################################################################

data = np.array([
    12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29
])

####################################################################################################
# One Sample Z Prop

oszp_data_a = RandState.binomial(n=1, p=.1, size=1000)
oszp_data_b = RandState.binomial(n=1, p=.1, size=1000)
oszp_data_c = RandState.binomial(n=1, p=.1, size=1000)
oszp_data_d = RandState.binomial(n=1, p=.1, size=1000)
oszp_data_e = RandState.binomial(n=1, p=.1, size=1000)

####################################################################################################
# One Sample Z

osz_data_a = RandState.normal(0.2, 1.0, 10)
osz_data_b = RandState.normal(0.0, 1.1, 101)
osz_data_c = RandState.normal(-5.0, 1.0, 3)
osz_data_d = RandState.normal(-0.2, 10.0, 10)
osz_data_e = np.array([
    -1.38921277, 0.84613978, -1.23715125, -0.19327593, -1.1869151,
    -0.27826107, -1.42672596, 0.31909449, -0.32760783, -0.67714162,
    -0.46351899, 0.77504551, -0.71160142, -0.23811822, -1.1091346,
    -0.06673534, 0.12672573, 2.25224209, 0.15327528, 0.16863875,
    0.93692492, -1.03549422, -1.60314673, -0.8463265, -1.13006108
])

####################################################################################################
# One Sample T

ost_data_a = RandState.normal(0.0, 1.0, 10)
ost_data_b = RandState.normal(0.0, 1.0, 100)
ost_data_c = RandState.normal(5.0, 1.0, 3)
ost_data_d = RandState.normal(-0.2, 1.0, 10)
ost_data_e = np.array([
    -2.43761614, -0.41811306, -0.69901818,  0.73337635,  2.38955199,
    -1.09788011, -0.27002167,  1.15445243, -0.91164842, -0.03091552,
    -0.66578974,  1.34411932, -0.49018175,  0.35946621,  0.31561145,
    0.37018058,  0.31960109, -0.02712077, -1.04763948,  0.14487796,
    2.03696463,  0.38568816, -1.06842843, -1.12761, -3.20686671
])

####################################################################################################
# Two Sample Z Prop

tszp_data_a = RandState.binomial(n=1, p=.1, size=1000), RandState.binomial(n=1, p=.1, size=1000)
tszp_data_b = RandState.binomial(n=1, p=.12, size=1000), RandState.binomial(n=1, p=.09, size=1000)
tszp_data_c = RandState.binomial(n=1, p=.5, size=10), RandState.binomial(n=1, p=.09, size=1000)
tszp_data_d = RandState.binomial(n=1, p=.1, size=10), RandState.binomial(n=1, p=.1, size=15)
tszp_data_e = (
    np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0
    ]),
    np.array([
        1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    ])
)

####################################################################################################
# Two Sample Z

tsz_data_a = RandState.normal(0.0, 1.0, 1000), RandState.normal(0.0, 1.0, 1000)
tsz_data_b = RandState.normal(5.0, 1.0, 10), RandState.normal(0.0, 1.0, 10)
tsz_data_c = RandState.normal(0.0, 6.0, 10), RandState.normal(1.0, 1.0, 10)
tsz_data_d = (
    np.array([
        0.51446116,  0.15483003, -0.37956859,  0.65118945,  0.50247888,
        -2.08047719,  0.17157756, -0.38133086,  1.10867976, -2.75840496
    ]),
    np.array([
        0.93601472, -0.77814049, -0.97060456, -1.27140527,  0.7671831,
        0.25756761, -0.39895448, -0.57098964,  0.08654137, -1.70676765
    ])
)

####################################################################################################
# Two Sample T

tst_data_a = RandState.normal(5.0, 5.0, 100), RandState.normal(0.0, 1.0, 100)
tst_data_b = RandState.normal(0.2, 10.0, 10), RandState.normal(0.0, 1.0, 10)
tst_data_c = RandState.normal(-2.0, 1.0, 10), RandState.normal(3.0, 7.0, 1000)
tst_data_d = RandState.normal(0.0, 10.0, 10), RandState.normal(1.1, 4.0, 1000)
tst_data_e = (
    np.array([
        0.03394189, 0.05784939, -0.84595132, 0.54908418, -0.44556832,
        -0.1073476, 0.30133315, 2.23448837, -1.53087909, 0.09569246
    ]),
    np.array([
        -2.19026959, -1.31884977, 0.10149457, -2.46556609, 1.24447421,
        1.37364603, 1.03016628, -0.78382578, 0.62710414, -0.71967579
    ])
)
