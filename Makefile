# C++ Compiler and Flags
CC      := g++
CFLAGS  := -std=c++14 -Wall -lrt

# CUDA Compiler and Flags
# Google Colab offers T4 GPUs which are the sm_75 compute capability. Since we had a
# a GPU on hand with sm_86 compute capability, we prioritized that if it exists
NCC     := nvcc
NLIB    := /usr/local/cuda/lib64
NFLAGS  := -std=c++14 -lrt -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75 -Xcompiler "-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS"

# Directories
G_SRC_DIR := ./src
G_INC_DIR := ./include

C_SRC_DIR := ./lib/viola-jones/src
C_INC_DIR := ./lib/viola-jones/include

# Gather GPU source files and convert to object files list
G_SRCS_C   := $(wildcard $(G_SRC_DIR)/*.c)
G_OBJS_C   := $(subst .c,.o, $(G_SRCS_C))

G_SRCS_CPP := $(wildcard $(G_SRC_DIR)/*.cpp)
G_OBJS_CPP := $(subst .cpp,.o, $(G_SRCS_CPP))

G_SRCS_CU := $(wildcard $(G_SRC_DIR)/*.cu)
G_OBJS_CU := $(subst .cu,.o, $(G_SRCS_CU))

# Gather CPU source files and convert to object files list
C_SRCS_C   := $(wildcard $(C_SRC_DIR)/*.c)
C_OBJS_C   := $(subst .c,.o, $(C_SRCS_C))

C_SRCS_CPP := $(wildcard $(C_SRC_DIR)/*.cpp)
C_OBJS_CPP := $(subst .cpp,.o, $(C_SRCS_CPP))

# Create GPU objs list total
G_OBJS    := $(G_OBJS_C) $(G_OBJS_CPP) $(G_OBJS_CU)

# Create CPU objs list total
C_OBJS    := $(C_OBJS_C) $(C_OBJS_CPP)

# Don't allow files in directory to be names clean or default or build
.PHONY: clean default build test perf

default: build test perf

build: build_gpu build_cpu

build_cpu: ./lib/viola-jones/main.cpp $(C_OBJS)
	@echo "Building CPU build..."
	$(CC) $(CFLAGS) -I $(C_INC_DIR) $^ -o facial_extract_cpu
	mkdir -p ./logs/cpu

build_gpu: main.cpp $(G_OBJS)
	@echo "Building GPU build..."
	$(NCC) $(NFLAGS) -I $(NLIB) -I $(G_INC_DIR) $^ -o facial_extract_gpu
	mkdir -p ./logs/gpu

$(G_SRC_DIR)/%.o: $(G_SRC_DIR)/%.cu $(G_INC_DIR)/%.cuh
	$(NCC) $(NFLAGS) -I $(G_INC_DIR) -I $(NLIB) -c $< -o $@

$(G_SRC_DIR)/%.o: $(G_SRC_DIR)/%.cpp $(G_INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(G_INC_DIR) -c $< -o $@

$(G_SRC_DIR)/%.o: $(G_SRC_DIR)/%.c $(G_INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(G_INC_DIR) -c $< -o $@

$(C_SRC_DIR)/%.o: $(C_SRC_DIR)/%.cpp $(C_INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(C_INC_DIR) -c $< -o $@

$(C_SRC_DIR)/%.o: $(C_SRC_DIR)/%.c $(C_INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(C_INC_DIR) -c $< -o $@

test: test_cpu test_gpu

test_cpu:
	@echo "Testing CPU runs on images in PGM_Images/..."
	@echo "Logs are stored in logs/cpu/"
	mkdir -p ./Detected_Images/cpu/
	./facial_extract_cpu -i PGM_Images/photo1_6_people.pgm -o ./Detected_Images/cpu/photo1_6_people.pgm > ./logs/cpu/photo1_6_people.log
	./facial_extract_cpu -i PGM_Images/photo2_4_people.pgm -o ./Detected_Images/cpu/photo2_4_people.pgm > ./logs/cpu/photo2_4_people.log
	./facial_extract_cpu -i PGM_Images/photo3_5_people.pgm -o ./Detected_Images/cpu/photo3_5_people.pgm > ./logs/cpu/photo3_5_people.log
	./facial_extract_cpu -i PGM_Images/photo4_7_people.pgm -o ./Detected_Images/cpu/photo4_7_people.pgm > ./logs/cpu/photo4_7_people.log
	./facial_extract_cpu -i PGM_Images/photo5_7_people.pgm -o ./Detected_Images/cpu/photo5_7_people.pgm > ./logs/cpu/photo5_7_people.log
	./facial_extract_cpu -i PGM_Images/photo6_4_people.pgm -o ./Detected_Images/cpu/photo6_4_people.pgm > ./logs/cpu/photo6_4_people.log
	./facial_extract_cpu -i PGM_Images/photo7_4_people.pgm -o ./Detected_Images/cpu/photo7_4_people.pgm > ./logs/cpu/photo7_4_people.log
	./facial_extract_cpu -i PGM_Images/photo8_4_people.pgm -o ./Detected_Images/cpu/photo8_4_people.pgm > ./logs/cpu/photo8_4_people.log
	./facial_extract_cpu -i PGM_Images/photo9_2_people.pgm -o ./Detected_Images/cpu/photo9_2_people.pgm > ./logs/cpu/photo9_2_people.log
	./facial_extract_cpu -i PGM_Images/photo10_5_people.pgm -o ./Detected_Images/cpu/photo10_5_people.pgm > ./logs/cpu/photo10_5_people.log

test_gpu:
	@echo "Testing GPU runs on images in PGM_Images/..."
	@echo "Logs are stored in logs/gpu/"
	mkdir -p ./Detected_Images/gpu/
	./facial_extract_gpu -i PGM_Images/photo1_6_people.pgm -o ./Detected_Images/gpu/photo1_6_people.pgm > ./logs/gpu/photo1_6_people.log
	./facial_extract_gpu -i PGM_Images/photo2_4_people.pgm -o ./Detected_Images/gpu/photo2_4_people.pgm > ./logs/gpu/photo2_4_people.log
	./facial_extract_gpu -i PGM_Images/photo3_5_people.pgm -o ./Detected_Images/gpu/photo3_5_people.pgm > ./logs/gpu/photo3_5_people.log
	./facial_extract_gpu -i PGM_Images/photo4_7_people.pgm -o ./Detected_Images/gpu/photo4_7_people.pgm > ./logs/gpu/photo4_7_people.log
	./facial_extract_gpu -i PGM_Images/photo5_7_people.pgm -o ./Detected_Images/gpu/photo5_7_people.pgm > ./logs/gpu/photo5_7_people.log
	./facial_extract_gpu -i PGM_Images/photo6_4_people.pgm -o ./Detected_Images/gpu/photo6_4_people.pgm > ./logs/gpu/photo6_4_people.log
	./facial_extract_gpu -i PGM_Images/photo7_4_people.pgm -o ./Detected_Images/gpu/photo7_4_people.pgm > ./logs/gpu/photo7_4_people.log
	./facial_extract_gpu -i PGM_Images/photo8_4_people.pgm -o ./Detected_Images/gpu/photo8_4_people.pgm > ./logs/gpu/photo8_4_people.log
	./facial_extract_gpu -i PGM_Images/photo9_2_people.pgm -o ./Detected_Images/gpu/photo9_2_people.pgm > ./logs/gpu/photo9_2_people.log
	./facial_extract_gpu -i PGM_Images/photo10_5_people.pgm -o ./Detected_Images/gpu/photo10_5_people.pgm > ./logs/gpu/photo10_5_people.log

perf:
	@echo "Parsing logs..."
	@echo "Plots generated in logs/plots/"
	python3 run_parser.py ./logs/cpu ./logs/gpu ./logs

clean:
	$(RM) -rf ./logs ./Detected_Images ./facial_extract* $(G_SRC_DIR)/*.o $(G_INC_DIR)/*.h.gch $(C_SRC_DIR)/*.o $(C_INC_DIR)/*.h.gch
