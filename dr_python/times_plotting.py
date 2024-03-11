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

import numpy as np
import json
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join, isdir
import re


class times_plots:
    def __init__(self, folder_path, agg=False):
        # Folder_path is the absolute path to a folder that
        # contains multiple subfolders of the form "sometext_(numberofagents)".
        # The subfolders include data from hypergradient runs with varying
        # numbers of agents/building.
        directories = [
            dir for dir in listdir(folder_path) if isdir(join(folder_path, dir))
        ]
        directories.sort(key=lambda f: int(re.sub("\D", "", f)))
        dir_paths = []
        for dir in directories:
            dir_paths.append(os.path.abspath(os.path.join(folder_path, dir)))
        self.dir_paths = dir_paths

        # List with the number of agents per simulation
        self.n_agents = list(map(lambda f: int(re.sub("\D", "", f)), directories))

        # Get number of runs
        self.n_runs = len(self.n_agents)

        # Attribute specifying whether the problem aggregative
        self.agg = agg

        # Load data
        self.load_times_data()

    def get_data_paths(self, path):
        # %% Get files
        onlyfiles = [
            f for f in listdir(path) if isfile(join(path, f)) if f.endswith(".json")
        ]
        onlyfiles.sort(key=lambda f: int(re.sub("\D", "", f)))
        filepaths = []
        for f in onlyfiles:
            filepaths.append(os.path.abspath(os.path.join(path, f)))
        return filepaths

    def load_times_data(self):
        inner_times = []
        outer_times = []
        # Loop over all simulations
        for ii, dir in enumerate(self.dir_paths):
            paths = self.get_data_paths(dir)
            inner_tmp = np.empty((self.n_agents[ii], 0))
            outer_tmp = np.empty(0)

            # Loop over all data from given simulation
            for path in paths:
                with open(path) as file:
                    data_dict = json.loads(file.read())

                # Read data and append
                data_inner = np.array(data_dict["vitimespnd"]) + np.array(
                    data_dict["vitimessens"]
                )
                # Here we include the time for aggregative s and y, if this time is attributed to the leader
                # if not(self.agg):
                #     data_outer = data_dict["leadtimes"]
                # else:
                #     data_outer = np.array(data_dict["leadtimes"]) + np.array(data_dict["leadaggeqt"]) + np.array(data_dict["leadaggsenst"])
                data_outer = data_dict["leadtimes"]

                inner_tmp = np.append(inner_tmp, data_inner, axis=1)
                outer_tmp = np.append(outer_tmp, data_outer)

            # Append completed simulation data
            inner_times.append(inner_tmp)
            outer_times.append(outer_tmp)

        self.inner_times = inner_times
        self.outer_times = outer_times

    def plot_leader_times(self, followers=False):
        avgs = np.zeros(self.n_runs)
        stds = np.zeros(self.n_runs)
        for ii in range(self.n_runs):
            avgs[ii] = np.average(self.outer_times[ii])
            stds[ii] = np.std(self.outer_times[ii])

        plt.errorbar(
            x=self.n_agents,
            y=avgs,
            yerr=stds,
            color="red",
            elinewidth=2,
            capsize=5,
            label="Leader",
        )

        if followers:
            avgs_fol = np.zeros(self.n_runs)
            stds_fol = np.zeros(self.n_runs)
            for ii in range(self.n_runs):
                avgs_fol[ii] = np.average(np.max(self.inner_times[ii], axis=0))
                stds_fol[ii] = np.std(np.max(self.inner_times[ii], axis=0))
            plt.errorbar(
                x=self.n_agents,
                y=avgs_fol,
                yerr=stds_fol,
                color="blue",
                elinewidth=2,
                capsize=5,
                label="Max Followers",
            )
            plt.legend()
        plt.yscale("log")
        plt.grid()
        plt.title("CPU time (per iteration)")
        plt.xlabel("Number of Agents")
        plt.ylabel("Time [s]")

    def plot_leader_times_shade(self, followers=True):
        avgs = np.zeros(self.n_runs)
        stds = np.zeros(self.n_runs)
        for ii in range(self.n_runs):
            avgs[ii] = np.average(self.outer_times[ii])
            stds[ii] = np.std(self.outer_times[ii])

        plt.plot(self.n_agents, avgs, color="red", linewidth=2, label="Leader")
        plt.fill_between(
            x=self.n_agents,
            y1=avgs - stds,
            y2=avgs + stds,
            color="red",
            alpha=0.3,
            edgecolor="None",
        )
        avgs_fol = np.zeros(self.n_runs)
        stds_fol = np.zeros(self.n_runs)
        for ii in range(self.n_runs):
            avgs_fol[ii] = np.average(np.max(self.inner_times[ii], axis=0))
            stds_fol[ii] = np.std(np.max(self.inner_times[ii], axis=0))
        plt.fill_between(
            x=self.n_agents,
            y1=avgs_fol - stds_fol,
            y2=avgs_fol + stds_fol,
            color="blue",
            alpha=0.3,
            edgecolor="None",
        )
        plt.plot(
            self.n_agents, avgs_fol, color="blue", linewidth=2, label="Max Follower"
        )
        plt.yscale("log")
        plt.legend(loc="center right")
        plt.grid()
        plt.xlabel("Number of Followers")
        plt.ylabel("Time [s]")
        plt.xticks(list(plt.xticks()[0]) + [3])
        plt.xlim([np.min(self.n_agents), 100])
        plt.ylim([1e-3, 4e-2])
        # plt.title("CPU time/iteration")
        # Get secondary axis
        ax = plt.gca()

        def ag2dim(x):
            return 96 * x + 48

        def dim2ag(x):
            return (x - 48) / 96

        secax = ax.secondary_xaxis("top", functions=(ag2dim, dim2ag))
        secax.set_xlabel("Problem Dimension")
        sec_ticks = 96 * np.array(list(plt.xticks()[0])) + 48
        secax.set_xticks(sec_ticks)

        plt.tight_layout()
        plt.show()
        plt.tight_layout()

    def plot_total_times(self):
        wallclock_avg = np.zeros(self.n_runs)
        wallclock_std = np.zeros(self.n_runs)
        virtual_avg = np.zeros(self.n_runs)
        virtual_std = np.zeros(self.n_runs)
        for ii in range(self.n_runs):
            # Wallclock time, i.e., each agent has their own CPU
            wallclock_avg[ii] = np.average(self.outer_times[ii]) + np.average(
                np.max(self.inner_times[ii], axis=0)
            )
            wallclock_std[ii] = np.std(
                self.outer_times[ii] + np.max(self.inner_times[ii], axis=0)
            )

            # Virtual time, i.e., total computation time
            virtual_avg[ii] = np.average(self.outer_times[ii]) + np.average(
                np.sum(self.inner_times[ii], axis=0)
            )
            virtual_std[ii] = np.std(
                self.outer_times[ii] + np.sum(self.inner_times[ii], axis=0)
            )

        plt.errorbar(
            x=self.n_agents,
            y=wallclock_avg,
            yerr=wallclock_std,
            color="blue",
            elinewidth=2,
            capsize=5,
            label="Distributed",
        )
        plt.errorbar(
            x=self.n_agents,
            y=virtual_avg,
            yerr=virtual_std,
            color="red",
            elinewidth=2,
            capsize=5,
            label="Serial",
        )
        plt.grid()
        plt.legend()
        plt.title("CPU time (per iteration)")
        plt.xlabel("Number of Followers")
        plt.ylabel("Time [s]")
        plt.yscale("log")

    def plot_total_times_shade(self):
        wallclock_avg = np.zeros(self.n_runs)
        wallclock_std = np.zeros(self.n_runs)
        virtual_avg = np.zeros(self.n_runs)
        virtual_std = np.zeros(self.n_runs)
        for ii in range(self.n_runs):
            # Wallclock time, i.e., each agent has their own CPU
            wallclock_avg[ii] = np.average(self.outer_times[ii]) + np.average(
                np.max(self.inner_times[ii], axis=0)
            )
            wallclock_std[ii] = np.std(
                self.outer_times[ii] + np.max(self.inner_times[ii], axis=0)
            )

            # Virtual time, i.e., total computation time
            virtual_avg[ii] = np.average(self.outer_times[ii]) + np.average(
                np.sum(self.inner_times[ii], axis=0)
            )
            virtual_std[ii] = np.std(
                self.outer_times[ii] + np.sum(self.inner_times[ii], axis=0)
            )

        plt.fill_between(
            x=self.n_agents,
            y1=wallclock_avg - wallclock_std,
            y2=wallclock_avg + wallclock_std,
            color="blue",
            alpha=0.3,
            edgecolor="None",
        )
        plt.plot(
            self.n_agents, wallclock_avg, color="blue", linewidth=2, label="Distributed"
        )

        plt.fill_between(
            x=self.n_agents,
            y1=virtual_avg - virtual_std,
            y2=virtual_avg + virtual_std,
            color="red",
            alpha=0.3,
            edgecolor="None",
        )
        plt.plot(self.n_agents, virtual_avg, color="red", linewidth=2, label="Serial")
        plt.grid()
        plt.legend(loc="center right")
        # plt.title("CPU time (per iteration)")
        plt.xlabel("Number of Followers")
        plt.ylabel("Time [s]")
        plt.yscale("log")
        plt.xticks(list(plt.xticks()[0]) + [3])
        plt.xlim([np.min(self.n_agents), 100])
        plt.tight_layout()
        # Secondary axis
        ax = plt.gca()

        def ag2dim(x):
            return 96 * x + 48

        def dim2ag(x):
            return (x - 48) / 96

        secax = ax.secondary_xaxis("top", functions=(ag2dim, dim2ag))
        secax.set_xlabel("Problem Dimension")
        sec_ticks = 96 * np.array(list(plt.xticks()[0])) + 48
        secax.set_xticks(sec_ticks)

        plt.tight_layout()
        plt.show()
