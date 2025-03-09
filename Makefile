# C++ Compiler and Flags
CC      := g++
CFLAGS  := -std=c++14 -Wall

# CUDA Compiler and Flags
# Google Colab offers T4 GPUs which are the sm_75 compute capability. Since we had a
# a GPU on hand with sm_86 compute capability, we prioritized that if it exists
NCC     := nvcc
NLIB    := /usr/local/cuda/lib64
NFLAGS  := -std=c++14 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75 -Xcompiler "-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS"

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
.PHONY: clean default build

default: build

build: build_gpu build_cpu

build_cpu: ./lib/viola-jones/main.cpp $(C_OBJS)
	$(CC) $(CFLAGS) -I $(C_INC_DIR) $^ -o facial_extract_cpu

build_gpu: main.cpp $(G_OBJS)
	$(NCC) $(NFLAGS) -I $(NLIB) -I $(G_INC_DIR) $^ -o facial_extract_gpu

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

clean:
	$(RM) -f ./facial_extract* $(G_SRC_DIR)/*.o $(G_INC_DIR)/*.h.gch $(C_SRC_DIR)/*.o $(C_INC_DIR)/*.h.gch
