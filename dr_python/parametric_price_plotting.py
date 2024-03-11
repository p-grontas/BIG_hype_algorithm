"""
   Copyright 2024 ETH Zurich, Panagiotis Grontas

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_parametric_price_problem(
    x, y, tsteps, n_ag, net_d_i, E_max_i, c0_min, c0_max, c1_min, c1_max, g_cap
):
    # x = Leader's Variable
    # y = Follower's Variable
    # %% Parse Solved Problem Data
    index_var = lambda ind: np.tile(
        np.concatenate(
            (
                np.full((ind - 1) * tsteps, False),
                np.full((tsteps,), True),
                np.full(((4 - ind) * tsteps,), False),
            )
        ),
        (n_ag,),
    )
    pi = y[index_var(1)].reshape((tsteps, n_ag), order="F")
    ei = y[index_var(2)].reshape((tsteps, n_ag), order="F")
    pci = y[index_var(3)].reshape((tsteps, n_ag), order="F")
    pdci = y[index_var(4)].reshape((tsteps, n_ag), order="F")
    # %% Do Plotting
    # Plot Agents' Profile
    steps_style = "steps-post"
    times = np.arange(tsteps)
    for ag in range(n_ag):
        plt.figure(ag)
        # Powers
        plt.subplot(211)
        plt.plot(times, pi[:, ag], drawstyle=steps_style, label="$p_i$", c="black")
        plt.plot(
            times,
            net_d_i[:tsteps, ag],
            drawstyle=steps_style,
            linestyle="--",
            label="$d_i - s_i$",
            c="red",
        )
        plt.xticks(times)
        plt.grid()
        plt.legend()
        # Batteries
        plt.subplot(212)
        plt.plot(times, ei[:, ag], drawstyle=steps_style, label="$e_i$", c="black")
        plt.plot(
            times,
            E_max_i[ag] * np.ones(tsteps),
            drawstyle=steps_style,
            linestyle="--",
            label="$e_{max}$",
            c="red",
        )
        plt.xticks(times)
        plt.grid()
        plt.legend()

    # %% Pricing map and overall power consumption
    c0 = x[:tsteps]
    c1 = x[tsteps : 2 * tsteps]
    plt.figure(n_ag)
    plt.subplot(211)
    prices = np.multiply(c1, pi.sum(axis=1)) + c0
    price_c0_max = np.multiply(c1, pi.sum(axis=1)) + c0_max
    price_c0_min = np.multiply(c1, pi.sum(axis=1)) + c0_min
    price_max = np.multiply(c1_max, pi.sum(axis=1)) + c0_max
    price_min = np.multiply(c1_min, pi.sum(axis=1)) + c0_min
    plt.plot(
        times, prices, drawstyle=steps_style, c="black", label="Price", linewidth=2
    )
    plt.plot(
        times,
        price_max,
        drawstyle=steps_style,
        c="red",
        linestyle="-",
        label="Price Max",
    )
    plt.plot(
        times,
        price_min,
        drawstyle=steps_style,
        c="blue",
        linestyle="-",
        label="Price Min",
    )
    plt.plot(
        times,
        price_c0_max,
        drawstyle=steps_style,
        c="red",
        linestyle="--",
        label="Price c_0 Max",
    )
    plt.plot(
        times,
        price_c0_min,
        drawstyle=steps_style,
        c="blue",
        linestyle="--",
        label="Price c_0 Min",
    )
    plt.xticks(times)
    plt.grid()
    plt.legend()
    plt.subplot(212)
    # plt.plot(
    #     times,
    #     g_cap,
    #     drawstyle=steps_style,
    #     linestyle="-",
    #     label="$g_i$",
    #     c="black",
    #     linewidth=2,
    # )
    plt.plot(
        times,
        net_d_i[:tsteps, :n_ag].sum(axis=1),
        drawstyle=steps_style,
        linestyle="--",
        label="$d_i - s_i$",
        c="red",
        linewidth=2,
    )
    plt.plot(
        times,
        pi.sum(axis=1),
        drawstyle=steps_style,
        label="$\sum_i p_i$",
        c="blue",
        linewidth=2,
    )
    plt.xticks(times)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # %% More Detailed Pricing Map
    plt.figure(n_ag + 1)
    # Marginal Prices c1
    plt.subplot(311)
    plt.plot(times, c1, drawstyle=steps_style, c="black", label="C1", linewidth=2)
    plt.plot(
        times,
        c1_max * np.ones(tsteps),
        drawstyle=steps_style,
        c="red",
        linestyle="--",
        label="C1_max",
        linewidth=2,
    )
    plt.plot(
        times,
        c1_min * np.ones(tsteps),
        drawstyle=steps_style,
        c="blue",
        linestyle="--",
        label="C1_min",
        linewidth=2,
    )
    plt.xticks(times)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Baseline Prices c0
    plt.subplot(312)
    plt.plot(times, c0, drawstyle=steps_style, c="black", label="C0", linewidth=2)
    plt.plot(
        times,
        c0_max * np.ones(tsteps),
        drawstyle=steps_style,
        c="red",
        linestyle="--",
        label="C0_max",
        linewidth=2,
    )
    plt.plot(
        times,
        c0_min * np.ones(tsteps),
        drawstyle=steps_style,
        c="blue",
        linestyle="--",
        label="C0_min",
        linewidth=2,
    )
    plt.xticks(times)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Power Consumption
    plt.subplot(313)
    # plt.plot(
    #     times,
    #     g_cap,
    #     drawstyle=steps_style,
    #     linestyle="-",
    #     label="$g_i$",
    #     c="black",
    #     linewidth=2,
    # )
    plt.plot(
        times,
        net_d_i[:tsteps, :n_ag].sum(axis=1),
        drawstyle=steps_style,
        linestyle="--",
        label="$d_i - s_i$",
        c="red",
        linewidth=2,
    )
    plt.plot(
        times,
        pi.sum(axis=1),
        drawstyle=steps_style,
        label="$\sum_i p_i$",
        c="blue",
        linewidth=2,
    )
    plt.xticks(times)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
