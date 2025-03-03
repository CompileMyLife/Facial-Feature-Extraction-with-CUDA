NVCC = nvcc
# Full path to the 64-bit cl.exe from Visual Studio Build Tools 2022, using forward slashes.
CCBIN = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64"
# GPU Address Sanitizer flag for CUDA 12.8
SANITIZER = -fsanitize=address
# Compiler flags for NVCC: C++14, debugging symbols (-g -G), sanitizer flag, and exception handling with deprecation suppression.
CFLAGS = -std=c++14 -g -G $(SANITIZER) -Xcompiler "/EHsc -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS"
# List all source files to compile with NVCC.
SRC = cuda_detect.cu main_cuda.cpp ../third_party/viola-jones/image.c ../third_party/viola-jones/haar.cpp ../third_party/viola-jones/rectangles.cpp
TARGET = face_detect_cuda.exe

all: $(TARGET)

$(TARGET):
	$(NVCC) -ccbin $(CCBIN) $(CFLAGS) -o $(TARGET) $(SRC) -I../third_party/viola-jones -I../include -Xlinker "/MACHINE:X64"

clean:
	del $(TARGET)
NVCC = nvcc
# Full path to the 64-bit cl.exe from Visual Studio Build Tools 2022, using forward slashes.
CCBIN = "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64"
# GPU Address Sanitizer flag for CUDA 12.8
SANITIZER = -fsanitize=address
# Compiler flags for NVCC: C++14, debugging symbols (-g -G), sanitizer flag, and exception handling with deprecation suppression.
CFLAGS = -std=c++14 -g -G $(SANITIZER) -Xcompiler "/EHsc -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS"
# List all source files to compile with NVCC.
SRC = cuda_detect.cu main_cuda.cpp ../third_party/viola-jones/image.c ../third_party/viola-jones/haar.cpp ../third_party/viola-jones/rectangles.cpp
TARGET = face_detect_cuda.exe

all: $(TARGET)

$(TARGET):
	$(NVCC) -ccbin $(CCBIN) $(CFLAGS) -o $(TARGET) $(SRC) -I../third_party/viola-jones -I../include -Xlinker "/MACHINE:X64"

clean:
	del $(TARGET)
