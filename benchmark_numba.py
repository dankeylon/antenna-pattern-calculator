import numba as nb
from numba import cuda
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import time as t
import warnings

from Antenna_Phased_Array import Array_Config, Array_2D

if False:
    dim = 128
    theta_l = np.linspace(-0.5*np.pi, 0.5*np.pi, dim)
    phi_l = np.linspace(-0.5*np.pi, 0.5*np.pi, dim)
    theta, phi = np.meshgrid(theta_l, phi_l)

    side_length = 30
    elements = Array_Config(side_length, side_length, 0.5, 0.5, 1).rectArray()

    k = 2 * np.pi
    radar_array = Array_2D(elements)


    # Test Single-Threaded CPU Case (No Numba)
    start = t.perf_counter()
    patt = radar_array.dF_dTheta(theta, phi, device = "cpu_single")
    end = t.perf_counter()

    print(f"CPU(1) Time: {end-start}s")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(theta_l, phi_l, np.abs(patt) )


    # Test Single-Threaded CPU Case with Numba
    start = t.perf_counter()
    patt = radar_array.dF_dTheta(theta, phi, device = "cpu_numba")
    end = t.perf_counter()

    print(f"CPU(Numba) Time: {end-start}s")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(theta_l, phi_l, np.abs(patt) )


    # Test Compiled Numba Multi-Threading
    start = t.perf_counter()
    patt = radar_array.dF_dTheta(theta, phi, device = "cpu_multi")
    end = t.perf_counter()

    print(f"CPU(M) Time: {end-start}s")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(theta_l, phi_l, np.abs(patt) )


    # Test Compiled Numba Cuda Multi-Threading
    start = t.perf_counter()
    patt = radar_array.dF_dTheta(theta, phi, device = "gpu")
    end = t.perf_counter()

    print(f"GPU Time: {end-start}s")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(theta_l, phi_l, np.abs(patt) )

    plt.show()


if False:
    # Iterate through a test matrix, run multiple iterations at each test point, average results
    pattern_resolution = np.array([64, 128, 256, 512, 1024])
    array_side_length =  np.array([2, 4, 8, 16, 32])
    assert len(pattern_resolution) == len(array_side_length)
    test_repeats = 5

    test_matrix_cpu       = np.zeros((len(pattern_resolution), len(array_side_length)))
    test_matrix_cpu_numba = np.zeros((len(pattern_resolution), len(array_side_length)))
    test_matrix_cpu_multi = np.zeros((len(pattern_resolution), len(array_side_length)))
    test_matrix_gpu       = np.zeros((len(pattern_resolution), len(array_side_length)))

    for rdx, res in enumerate(pattern_resolution):
        k = 2 * np.pi
        theta_l = np.linspace(-0.5*np.pi, 0.5*np.pi, pattern_resolution[rdx])
        phi_l   = np.linspace(-0.5*np.pi, 0.5*np.pi, pattern_resolution[rdx])
        theta, phi = np.meshgrid(theta_l, phi_l)

        for ldx, length in enumerate(array_side_length):
            elements = Array_Config(array_side_length[ldx], array_side_length[ldx], 0.5, 0.5, 1).rectArray()
            radar_array = Array_2D(elements)

            test_point_score_cpu = 0
            test_point_score_cpu_numba = 0
            test_point_score_cpu_multi = 0
            test_point_score_gpu = 0

            print(f"Beginning Test Point: res={pattern_resolution[rdx]}, length={array_side_length[ldx]}")
            for test_idx in range(0, test_repeats):
                    
                # Test CPU (Single, No Numba)
                start = t.perf_counter()
                patt = radar_array.dF_dTheta(theta, phi, device = "cpu_single")
                end = t.perf_counter()
                test_point_score_cpu += end-start

                # Test CPU (Single, Numba)
                start = t.perf_counter()
                patt = radar_array.dF_dTheta(theta, phi, device = "cpu_numba")
                end = t.perf_counter()
                test_point_score_cpu_numba += end-start

                # Test CPU (Multi, Numba)
                start = t.perf_counter()
                patt = radar_array.dF_dTheta(theta, phi, device = "cpu_multi")
                end = t.perf_counter()
                test_point_score_cpu_multi += end-start

                # Test GPU
                start = t.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    patt = radar_array.dF_dTheta(theta, phi, device = "gpu")
                end = t.perf_counter()
                test_point_score_gpu += end-start

            test_matrix_cpu[rdx, ldx] = round(test_point_score_cpu / test_repeats, 6)
            test_matrix_cpu_numba[rdx, ldx] = round(test_point_score_cpu_numba / test_repeats, 6)
            test_matrix_cpu_multi[rdx, ldx] = round(test_point_score_cpu_multi / test_repeats, 6)
            test_matrix_gpu[rdx, ldx] = round(test_point_score_gpu / test_repeats, 6)

    print("Writing Files")
    np.savetxt("test_matrix_cpu.csv", test_matrix_cpu)
    np.savetxt("test_matrix_cpu_numba.csv", test_matrix_cpu_numba)
    np.savetxt("test_matrix_cpu_multi.csv", test_matrix_cpu_multi)
    np.savetxt("test_matrix_gpu.csv", test_matrix_gpu)


if True:
    pattern_resolution = np.array([64, 128, 256, 512, 1024])
    array_side_length =  np.array([2, 4, 8, 16, 32])

    test_matrix_cpu       = np.loadtxt("test_matrix_cpu.csv")
    test_matrix_cpu_numba = np.loadtxt("test_matrix_cpu_numba.csv")
    test_matrix_cpu_multi = np.loadtxt("test_matrix_cpu_multi.csv")
    test_matrix_gpu       = np.loadtxt("test_matrix_gpu.csv")


    def plot_stuff(test_mat, name, ax = None):

        if ax is None:
            __, ax = plt.subplots(1,1)

        plot_array = np.array([test_mat[idx, idx] for idx in range(0, len(test_mat))])
        ax.plot(plot_array, label=name)
        ax.legend()

        return ax
    
    ax = plot_stuff(test_matrix_cpu, "CPU")
    plot_stuff(test_matrix_cpu_numba, "CPU-Numba", ax = ax)
    plot_stuff(test_matrix_cpu_multi, "CPU-Multi", ax = ax)
    plot_stuff(test_matrix_gpu, "GPU", ax = ax)

    plt.show()








