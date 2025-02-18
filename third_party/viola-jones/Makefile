# Compiler and flags
GCC ?= g++
CFLAGS := -Wall -Wextra -pedantic
LDFLAGS :=

# Source and header files (using wildcards)
SRC_C := $(wildcard *.c)
SRC_CPP := $(wildcard *.cpp)
HEADERS := $(wildcard *.h)

# Object files (derived from source files)
OBJ_C := $(SRC_C:.c=.o)
OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ := $(OBJ_C) $(OBJ_CPP)

# Target executable
TARGET := vj

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJ)
	$(GCC) -o $@ $^ $(LDFLAGS)

# Compile C files
%.o: %.c $(HEADERS)
	$(GCC) $(CFLAGS) -c $< -o $@

# Compile C++ files
%.o: %.cpp $(HEADERS)
	$(GCC) $(CFLAGS) -c $< -o $@

# Run
run: $(TARGET)
	./$(TARGET)

# Clean up (Windows-specific, using cmd /c and rm)
clean:
	cmd /c "rm -f $(TARGET) $(OBJ) Output.pgm"

# Phony targets
.PHONY: all run clean