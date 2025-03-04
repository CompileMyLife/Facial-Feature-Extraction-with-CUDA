CC      := g++
# Compiler flags for NVCC: C++14, debugging symbols (-g -G), and exception handling with deprecation suppression.
CFLAGS  := -std=c++14 -Wall -g

# CUDA Compiler and Flags
NCC     := nvcc
NLIB    := /usr/local/cuda/lib64
#NFLAGS  := -std=c++14 -arch=sm_75 -g -G # T4 GPU available on Google Colab
NFLAGS  := -std=c++14 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_75,code=sm_75 -g -G -Xcompiler "-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS"

# Directories
SRC_DIR := ./src
INC_DIR := ./include

SRCS_C   := $(wildcard $(SRC_DIR)/*.c)
OBJS_C   := $(subst .c,.o, $(SRCS_C))

SRCS_CPP := $(wildcard $(SRC_DIR)/*.cpp)
OBJS_CPP := $(subst .cpp,.o, $(SRCS_CPP))

SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
OBJS_CU := $(subst .cu,.o, $(SRCS_CU))

OBJS    := $(OBJS_C) $(OBJS_CPP) $(OBJS_CU)

# Not associated with files
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
