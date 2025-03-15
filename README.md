# Facial Feature Extraction (Viola-Jones Algorithm)

This project implements a basic version of the Viola-Jones object detection algorithm for facial feature extraction. It reads a PGM (Portable Graymap) image, processes it, and outputs a modified PGM image.

## Prerequisites

Before you can build and run the code, you need the following:

1.  **A C++ Compiler (GNU C++ Compiler):** You'll need a C++ compiler and build tools.

2.  **`make` Utility:** Verify that it's installed and in your PATH by opening a command prompt or PowerShell and typing `make --version`. You should see version information.

3.  **IrfanView (or another PGM viewer):** To view the PGM images, it's recommended to install IrfanView. Download it from [https://www.irfanview.com/](https://www.irfanview.com/). This program can also be used to convert between JPEG/PNG and PGM formats.

4. **Python Virtual Environment with certain modules** The modules required are in requirements.txt

## Building the Code

**Build with `make`:** Use the provided `Makefile` to build the project:

    ```bash
    make
    ```

    This will create an executable file named `facial_extract_cpu` and `facial_extract_gpu`. Since the Makefile is automated, the testing of our dataset across both executables are logged in `logs/cpu` and `logs/gpu`. The detected (drawn rectangles on candidates) images are located in `Detected_Images/`. The data analysis script generates plots in `logs/plots`

## Reference
## Reference
* [Viola-Jones-cpp](https://github.com/dev7saxena/Viola-Jones-cpp/tree/master)
## License
Please refer to the MIT license attached to this repository.
