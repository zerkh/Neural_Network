# Executable
EXE    = sentiment
	
# Compiler, Linker Defines
CC      = g++ -g
CFLAGS  = -Wall -O2 

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cc
	echo compiling...$<
	$(CC) $(CFLAGS) -c $<
# -o $@
# Source and Object files
SRC    = $(wildcard *.cpp)
OBJ    = $(patsubst %.cpp, %.o, $(SRC))

# Link all Object Files with external Libraries into Binaries
$(EXE): $(OBJ)
	echo linking...
	$(CC) $(CFLAGS) $(OBJ) -o $(EXE) -lz

.PHONY: clean
clean:
	 -rm -f core *.o

