import collections
import matplotlib.pyplot as plt
import os
import numpy as np
import helper.misc as hmisc

"""
Yeah simple class..Physlog cool
"""


class ReadPhyslog:
    """
    Reads the physlog.. Parses it upon initialization of the object

    Comment from the Philips Community:
    https://community.mr-paradise.com/t/physlog-markers/66

    Trigger_ECG = 0x01
    Trigger_PPU = 0x02
    Trigger_RESP = 0x04
    Measurement = 0x08
    Scan_begin = 0x10
    Scan_end = 0x20
    """
    sample_rate = 1.6 * 10 ** -3
    # sample_rate = 2 * 10 ** -3
    stop_mark = '0020'
    start_mark = '0010'

    def __init__(self, file):
        self.file = file
        self.file_lines = self.read_file()
        self.phys_log, self.columns = self.parse_phys_log()
        self.phys_log_dict = self.convert_to_dict()
        n_points = len(self.phys_log[0])
        self.phys_log_time = np.arange(n_points) * self.sample_rate

    def convert_to_dict(self):
        content_dict = {}
        for ii, icol in enumerate(self.columns):
            content_dict.setdefault(icol, [])
            data_vector = self.phys_log[ii]
            content_dict[icol].extend(data_vector)
        return content_dict

    @staticmethod
    def _isdigit(xstr):
        if isinstance(xstr, str):
            if xstr[0] in ('-', '+'):
                return xstr[1:].isdigit()
            return xstr.isdigit()
        elif isinstance(xstr, int) or isinstance(xstr, float):
            return True
        else:
            print("Unknown dtype ", type(xstr), xstr)

    def read_file(self, file=None):
        if file is None:
            file = self.file

        with open(file, 'r') as f:
            file_lines = f.readlines()

        return file_lines

    def parse_phys_log(self, file_lines=None):
        if file_lines is None:
            file_lines = self.file_lines

        return_column = None
        phys_log = []
        for i_line in file_lines:
            if i_line.startswith('##'):
                print(i_line)
            elif i_line.startswith('#'):
                column = i_line.strip()[2:].split()
                if len(column):
                    return_column = column
                    print(' Column s ', return_column)
            else:
                phys_line = i_line.strip().split()
                # Process everything as an int.. except the last (`mark`) columns
                phys_line[:-1] = [int(x) for x in phys_line[:-1]]
                phys_log.append(phys_line)

        phys_log = hmisc.change_list_order(phys_log)
        return phys_log, return_column

    def visualize_label(self, label):
        ppu_data = self.phys_log_dict[label]
        plot_max = np.abs(ppu_data).max()
        mark_counter = collections.Counter(self.phys_log_dict['mark'])
        fig, ax = plt.subplots(1)
        fig.suptitle(os.path.basename(self.file) + '  ' + str(label))
        ax.plot(self.phys_log_time, ppu_data)
        ax.set_ylim(-plot_max, plot_max)
        ignored_keys = ["0000", "0002"]
        for i_key, i_mark in mark_counter.items():
            if i_key not in ignored_keys:
                mark_index = np.argwhere([x == i_key for x in self.phys_log_dict['mark']])
                mark_index = mark_index.ravel()
                for i_mark_index in mark_index:
                    ax.vlines(x=self.phys_log_time[i_mark_index], ymin=-plot_max, ymax=plot_max, color='k', alpha=0.5)
                    ax.text(x=self.phys_log_time[i_mark_index], y=int(0.9 * plot_max), s=i_key)
        return fig


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import collections
    import os
    dir_phys_log = '/home/bugger/Documents/data/7T/cardiac'
    phys_log_file_name = 'SCANPHYSLOG_ca_01122021_1019026_17_2_transverse_retro_radialV4.log'
    phys_log_file = os.path.join(dir_phys_log, phys_log_file_name)

    phys_obj = ReadPhyslog(phys_log_file)

    ppu_data = phys_obj.phys_log_dict['ppu']
    plot_max = 7000
    fig, ax = plt.subplots(1)
    ax.plot(ppu_data)
    ax.set_ylim(-plot_max, plot_max)
    mark_counter = collections.Counter(phys_obj.phys_log_dict['mark'])
    # These are too common.
    # 0000 - normal data entry
    # 0002 - cardiac trigger
    ignored_keys = ["0000", "0002"]
    for i_key, i_mark in mark_counter.items():
        if i_key not in ignored_keys:
            mark_index = np.argwhere([x == i_key for x in phys_obj.phys_log_dict['mark']])
            mark_index = mark_index.ravel()
            for i_mark_index in mark_index:
                ax.vlines(x=i_mark_index, ymin=-plot_max, ymax=plot_max, color='k', alpha=0.5)
                ax.text(x=i_mark_index, y=int(0.9*plot_max), s=i_key)


