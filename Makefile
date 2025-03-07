# C++ Compiler and Flags
CC      := g++
CFLAGS  := -std=c++14 -Wall -g

# CUDA Compiler and Flags
# Google Colab offers T4 GPUs which are the sm_75 compute capability. Since we had a
# a GPU on hand with sm_86 compute capability, we prioritized that if it exists
NCC     := nvcc
NLIB    := /usr/local/cuda/lib64
NFLAGS  := -std=c++14 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75 -g -G -Xcompiler "-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS"

# Directories
SRC_DIR := ./src
INC_DIR := ./include

# Gather source files and convert to object files list
SRCS_C   := $(wildcard $(SRC_DIR)/*.c)
OBJS_C   := $(subst .c,.o, $(SRCS_C))

SRCS_CPP := $(wildcard $(SRC_DIR)/*.cpp)
OBJS_CPP := $(subst .cpp,.o, $(SRCS_CPP))

SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
OBJS_CU := $(subst .cu,.o, $(SRCS_CU))

OBJS    := $(OBJS_C) $(OBJS_CPP) $(OBJS_CU)

# Don't allow files in directory to be names clean or default or build
.PHONY: clean default build

default: build

build: main.cpp $(OBJS)
	$(NCC) $(NFLAGS) -I $(INC_DIR) $^ -o facial_extract

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NCC) $(NFLAGS) -I $(INC_DIR) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(INC_DIR) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(INC_DIR) -c $< -o $@

clean:
	$(RM) -f ./facial_extract src/*.o include/*.h.gch
