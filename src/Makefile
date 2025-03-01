NVCC = nvcc
SRC = cuda_detect.cu main_cuda.cpp
TARGET = face_detect_cuda

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC) -I../third_party/viola-jones -I../include

clean:
	del $(TARGET).exe
