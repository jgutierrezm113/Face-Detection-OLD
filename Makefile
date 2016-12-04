# environment
#SM := 35

# compilers
#CUDA_PATH ?= $(shell test -d /shared/apps/cuda7.0 && echo /shared/apps/cuda7.0)

#ifeq ($(CUDA_PATH),)
#CUDA_PATH := $(shell test -d /usr/local/cuda-7.0 && echo /usr/local/cuda-7.0)
#endif 

GCC := g++
#NVCC := $(CUDA_PATH)/bin/nvcc
#MPI  = mpiCC

# libraries
#CUDA_LIB_PATH := $(CUDA_PATH)/lib

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
#GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
#LIB_FLAGS := -lcudadevrt -lcudart
#ifeq ($(OS), DARWIN)
#	CCFLAGS := -stdlib=libstdc++
#else
#	CCFLAGS := -O3
#endif
#NVCCFLAGS :=
#GccFLAGS = -fopenmp -O3 
#MPIFLAGS = -Wno-deprecated

CaffeLocation = /home/julian/Downloads/caffe
CaffeLIB = -L$(CaffeLocation)/build/lib
CaffeINC = -I$(CaffeLocation)/include/

NetLocation = /home/julian/HCC/Project/Codes/face-detection/net
NetLIB = -L$(NetLocation)

GccFLAGS =  -pthread -std=c++11 -O3
GccLibs = $(CaffeLIB) $(CaffeINC) $(NetLIB)

GccLinkFLAGS = -lprotobuf -lglog `pkg-config opencv --cflags --libs` -lboost_system -lcaffe -lnet

debug: GccFLAGS += -DDEBUG -g -Wall
debug: all

# The build target executable:
TARGET = face_detector

all: build

build: $(TARGET)

# Create executable
$(TARGET): $(TARGET).cpp detector/detector.cpp net/libnet.so matlab/onetInput
	$(GCC) $(GccLibs) -Wl,-rpath=$(NetLocation) $< detector/detector.cpp -o $@ $(GccFLAGS) $(GccLinkFLAGS)

# Create Shared library for net objects
net/libnet.so: net/bnet.o net/pnet.o net/rnet.o net/onet.o
	$(GCC) $(CaffeINC) $(GccFLAGS) -shared $< net/pnet.o net/rnet.o net/onet.o -o $@

net/bnet.o: net/bnet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

net/pnet.o: net/pnet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

net/rnet.o: net/rnet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

net/onet.o: net/onet.cpp
	$(GCC) $(CaffeINC) $(GccFLAGS) -c -fpic $< -o $@

matlab/onetInput: matlab/onetInput.cpp
	matlab -nodisplay -nosplash -nodesktop -r "mex -v -client engine matlab/onetInput.cpp;exit;"
	mv onetInput matlab/
        
clean:
	$(RM) $(TARGET) *.o net/*.so net/*.o *.tar* *.core* matlab/onetInput
        
run:
	./$(TARGET) test1.jpg
