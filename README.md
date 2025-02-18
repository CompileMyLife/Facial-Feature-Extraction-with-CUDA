# Facial Feature Extraction (Viola-Jones Algorithm)

This project implements a basic version of the Viola-Jones object detection algorithm for facial feature extraction. It reads a PGM (Portable Graymap) image, processes it, and outputs a modified PGM image.

## Prerequisites

Before you can build and run the code, you need the following:

1.  **A C++ Compiler (MinGW-w64):** You'll need a C++ compiler and build tools.  MinGW-w64 is recommended on Windows.  Download and install it from [https://www.mingw-w64.org/](https://www.mingw-w64.org/). During installation, make sure to add the `bin` directory of your MinGW-w64 installation to your system's `PATH` environment variable.  The recommended variant is `x86_64-posix-seh`. A suitable installer can often be found on SourceForge, such as [https://sourceforge.net/projects/mingw-w64/files/](https://sourceforge.net/projects/mingw-w64/files/). Look for an archive named something like `x86_64-13.2.0-release-posix-seh-ucrt-rt_v11-rev1.7z` (the version numbers might be different). Extract the archive to a directory *without spaces* in the path (e.g., `C:\mingw64`).

2.  **`make` Utility:** The `make` utility is usually included with MinGW-w64.  Verify that it's installed and in your PATH by opening a command prompt or PowerShell and typing `make --version`. You should see version information.

3.  **IrfanView (or another PGM viewer):** To view the PGM images, it's recommended to install IrfanView. Download it from [https://www.irfanview.com/](https://www.irfanview.com/). This program can also be used to convert between JPEG/PNG and PGM formats.

## Building the Code

1.  **Navigate to the Project Directory:** Open a command prompt or PowerShell window and navigate to the `third_party/viola-jones` subdirectory within the project:

    ```bash
    cd <your_project_path>/Facial-Feature-Extraction-with-CUDA/third_party/viola-jones
    ```
    Replace `<your_project_path>` with the actual path where you cloned or downloaded the repository.  For example, if you cloned the repository to your Desktop, the path might be:
    ```bash
    cd C:\Users\YourName\Desktop\Facial-Feature-Extraction-with-CUDA\third_party\viola-jones
    ```

2.  **Build with `make`:** Use the provided `Makefile` to build the project:

    ```bash
    make
    ```

    This will create an executable file named `vj` (or `vj.exe`).

## Running the Code

1.  **Prepare your Input Image:**
    *   The program is currently configured to process PGM images.
    *   If you have a JPEG or PNG image, use IrfanView (or another tool like ImageMagick) to convert it to PGM format.  In IrfanView, open the image, go to "File" -> "Save As...", choose "PGM" as the output format, and save the file.  Make sure it's saved in the `viola-jones` directory.
    * **Replace `Face.pgm`:**  The example uses `Face.pgm`. Rename or move your desired input image to overwrite the included `Face.pgm` file within the `viola-jones` directory. *Alternatively*, you can modify the source code (likely in `main.cpp`) to accept a filename as a command-line argument, or to read from a different hardcoded filename.  *The easiest method is simply to replace the provided `Face.pgm`.*

2.  **Execute the Program:** Run the program using the `make run` command:

    ```bash
    make run
    ```

    This executes the `vj` program.

3. **View the Output**

    *   The program will create an output image named `Output.pgm` in the same directory.
    *   Open `Output.pgm` in IrfanView (or your chosen PGM viewer) to see the results.

## Cleaning Up

To remove the compiled files (executable and object files), run:

```bash
make clean