#!/usr/bin/env python

"""
Simple script for downloading trained Gaussian Processes
for use with the cross-section evaluation code.
Compatible with Python 2 and 3.

Usage:
  ./download_gprocs.py <directory-for-xsec-data>
"""


# Import packages
from __future__ import print_function
import os
import sys
import time
import errno
import tarfile

try:  # if python3
    from urllib.request import urlretrieve
except ImportError:  # if python2
    from urllib import urlretrieve


# Each download is specified as a (url,file_name) tuple in the list below.
DOWNLOAD_LIST = [
    ("https://github.com/jeriek/xstest/releases/download/0.0.1/data.tar.gz",
     "data.tar.gz")
]


def main():
    """
    Download all files in DOWNLOAD_LIST.
    """

    # If argument is given, use this at download directory
    # If no argument, specify current working directory as default

    # TODO: check if multiple arguments are given, raise error

    try:
        xsec_data_dir = sys.argv[1]
        print("Setting download directory for the data as specified:\n",
              xsec_data_dir)
    except IndexError:
        xsec_data_dir = os.getcwd()
        print("Setting download directory for the data to the current working "
              "directory:\n", xsec_data_dir)
    data_init_filepath = os.path.join(xsec_data_dir, '__init__.py')

    # Create data dir and/or init file if not existing
    if not os.path.exists(xsec_data_dir):
        try:
            # Create data directory and empty init file
            mkdir_p(xsec_data_dir)
            open(data_init_filepath, 'a').close()
        except OSError:
            raise
    elif not os.path.exists(data_init_filepath):
        try:
            # Create empty init file
            open(data_init_filepath, 'a').close()
        except OSError:
            raise

    # Print a new line
    print()

    for url, file_name in DOWNLOAD_LIST:
        print("-- Downloading file", file_name, "from\n", url, ":\n")
        # Force writing to terminal; otherwise stdout is buffered first
        sys.stdout.flush()

        # Download compressed file to a temporary location (since
        # filename=None)
        # @todo Put this in a try,except block to catch download errors
        tmp_filename, _ = urlretrieve(
            url, filename=None, reporthook=download_progress_hook)

        print("\n\n  ... download completed")
        print("  ... starting file extraction (", tmp_filename, ")")
        sys.stdout.flush()
        tar = tarfile.open(tmp_filename, 'r:gz')
        tar.extractall(xsec_data_dir)
        tar.close()
        print("  ... file extraction completed.")
        sys.stdout.flush()

    print()
    print("All downloads completed. Data files stored in",
          xsec_data_dir, "\b.")
    print()


def mkdir_p(path):
    """
    Safely make new directories recursively, as with "mkdir -p".
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def download_progress_hook(count, block_size, total_size):
    """
    Visualise download progress.
    Used with the urllib.urlretrieve reporthook.
    """

    # Compute time since start of download
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    # Compute combined size of downloaded chunks, given in MB
    progress_size = float(count * block_size)
    # Compute download speed in KB/s
    speed = int(progress_size / (1024. * duration))

    # When the HTTP response has 'Content-Length' in the header, display
    # progress bar and percentage of download completion. Else, when
    # 'Transfer-Encoding: chunked' is used in the header, no information
    # about total size is available, so only display accumulated
    # download size.
    if total_size > 0:
        percent = min(int(count*block_size*100/total_size), 100)
        # Setup toolbar
        toolbar_width = 18
        progress_symbol = "/"
        n_progress_lines = int(percent*toolbar_width/100.)

        # Output to screen: progress bar, some useful info
        # (\r returns cursor to start of line to overwrite old info,
        # \b backspace, \t tab)
        sys.stdout.write(
            "\r[%s] \t\b\b %s%%  --  %.1f MB, %.1f s, %d KB/s"
            % (progress_symbol*n_progress_lines
               + " " * (toolbar_width-n_progress_lines),
               percent,
               progress_size / (1024. * 1024.),
               duration,
               speed)
        )
        # Force writing to terminal; otherwise stdout is buffered first
        sys.stdout.flush()

    else:
        sys.stdout.write(
            "\r  ... downloaded %.1f MB in %.1f s, avg speed: %d KB/s"
            % (progress_size / (1024. * 1024.),
               duration,
               speed)
        )
        sys.stdout.flush()


# When the code is executed as a script, run the following.
if __name__ == '__main__':
    main()