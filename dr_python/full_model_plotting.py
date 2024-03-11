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


def plot_full_model(
    x, y, tsteps, n_ag, net_d_i, E_max_i, c0_min, c0_max, c1, g_cap, r, flex_perc=None
):
    c_0 = x[:tsteps]
    alpha = x[tsteps:]
    index_var = lambda ind: np.tile(
        np.concatenate(
            (
                np.full((ind - 1) * tsteps, False),
                np.full((tsteps,), True),
                np.full(((6 - ind) * tsteps,), False),
            )
        ),
        (n_ag,),
    )
    pi = y[index_var(1)].reshape((tsteps, n_ag), order="F")
    yi = y[index_var(2)].reshape((tsteps, n_ag), order="F")
    ei = y[index_var(3)].reshape((tsteps, n_ag), order="F")
    pci = y[index_var(4)].reshape((tsteps, n_ag), order="F")
    pdci = y[index_var(5)].reshape((tsteps, n_ag), order="F")
    ki = y[index_var(6)].reshape((tsteps, n_ag), order="F")
    # %% Do Plotting
    # Plot Agents' Profile
    steps_style = "steps"
    times = np.arange(tsteps)
    for ag in range(n_ag):
        plt.figure(ag)
        # plt.gcf().suptitle("Prosumer " + str(ag + 1))
        # Powers
        plt.subplot(311)
        plt.plot(
            times,
            pi[:, ag],
            drawstyle=steps_style,
            label="$p_i$",
            c="black",
            linewidth=2,
        )
        plt.plot(
            times,
            net_d_i[:tsteps, ag],
            drawstyle=steps_style,
            linestyle="--",
            label="$d_i - s_i$",
            c="red",
            linewidth=2,
        )
        plt.xticks(times)
        plt.xlim([0, tsteps - 1])
        plt.gca().set_ylim(bottom=0)
        plt.grid()
        plt.legend()
        plt.ylabel(r"$p_i$ [kW]")
        plt.gca().set_title("Prosumer " + str(ag + 1))

        # Batteries
        plt.subplot(312)
        plt.plot(
            times,
            ei[:, ag],
            drawstyle=steps_style,
            label="$e_i$",
            c="black",
            linewidth=2,
        )
        plt.plot(
            times,
            E_max_i[ag] * np.ones(tsteps),
            drawstyle=steps_style,
            linestyle="--",
            label="$e_{max}$",
            c="red",
            linewidth=2,
        )
        plt.xlim([0, tsteps - 1])
        plt.ylim([0, E_max_i[ag] + 5])
        plt.xticks(times)
        plt.grid()
        plt.legend()
        plt.ylabel(r"$e_i$ [kWh]")
        if not (flex_perc is None):
            flex = r * flex_perc[ag]
        else:
            flex = r

        plt.subplot(313)
        plt.plot(
            times,
            ki[:, ag],
            drawstyle=steps_style,
            label="$k_i$",
            c="green",
            linewidth=2,
        )
        plt.plot(
            times,
            yi[:, ag],
            drawstyle=steps_style,
            label="$y_i$",
            c="blue",
            linewidth=2,
        )
        plt.plot(
            times,
            flex[:tsteps],
            drawstyle=steps_style,
            linestyle="--",
            label=r"$ \hat{\zeta}_{i} r $",
            c="black",
            linewidth=2,
        )
        plt.xlim([0, tsteps - 1])
        plt.xticks(times)
        plt.grid()
        plt.legend()
        plt.ylabel("Flexibility [kW]")
        plt.xlabel("Time of day [h]")
        plt.tight_layout()
        plt.show()

    # %% Pricing map and overall power consumption
    plt.figure(n_ag)
    plt.subplot(311)
    prices = np.multiply(c1, pi.sum(axis=1)) + c_0
    price_max = np.multiply(c1, pi.sum(axis=1)) + c0_max
    price_min = np.multiply(c1, pi.sum(axis=1)) + c0_min
    plt.plot(
        times, prices, drawstyle=steps_style, c="black", label="Price", linewidth=2
    )
    plt.plot(
        times,
        price_max,
        drawstyle=steps_style,
        c="red",
        linestyle="--",
        label="Price Max",
        linewidth=2,
    )
    plt.plot(
        times,
        price_min,
        drawstyle=steps_style,
        c="blue",
        linestyle="--",
        label="Price Min",
        linewidth=2,
    )
    plt.xlim([0, tsteps - 1])
    plt.ylabel("Pricing map [CHF/kWh]")
    plt.xticks(times)
    plt.grid()
    plt.legend()

    plt.subplot(312)
    plt.plot(
        times,
        100 * alpha,
        drawstyle=steps_style,
        c="black",
        label=r"$ \alpha $",
        linewidth=2,
    )
    plt.ylabel("Distribution Share [\%]")
    plt.xlim([0, tsteps - 1])
    plt.xticks(times)
    plt.grid()
    plt.legend()

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
        label="$\sum_i d_i - s_i$",
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
    plt.xlim([0, tsteps - 1])
    plt.xticks(times)
    plt.ylabel("Overall Power [kW]")
    plt.xlabel("Time of day [h]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(n_ag + 1)
    plt.subplot(211)
    plt.plot(
        times,
        ki.sum(axis=1),
        drawstyle=steps_style,
        label="$k_i$",
        c="green",
        linewidth=2,
    )
    plt.plot(
        times,
        yi.sum(axis=1),
        drawstyle=steps_style,
        label="$y_i$",
        c="blue",
        linewidth=2,
    )
    plt.plot(
        times,
        r[:tsteps],
        drawstyle=steps_style,
        linestyle="--",
        label="r",
        c="black",
        linewidth=2,
    )
    plt.xlim([0, tsteps - 1])
    plt.xticks(times)
    plt.grid()
    plt.legend()
    plt.ylabel("Flexibility Provided [kW]")

    plt.subplot(212)
    plt.plot(
        times,
        g_cap - np.minimum(0, r),
        drawstyle=steps_style,
        linestyle="-",
        label="$g_i - \min(0, r)$",
        c="black",
        linewidth=2,
    )
    plt.plot(
        times,
        (pi + yi - ki).sum(axis=1),
        drawstyle=steps_style,
        label="$\sum_i p_i + y_i - k_i$",
        c="blue",
        linewidth=2,
    )
    plt.xlim([0, tsteps - 1])
    plt.xticks(times)
    plt.grid()
    plt.legend()
    plt.ylabel("Grid Capacity [kW]")
    plt.xlabel("Time of day [h]")
    plt.tight_layout()
    plt.show()
