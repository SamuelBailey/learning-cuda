HIPIFY=hipify-clang
CXX=hipcc

CXX_FLAGS = -O3 -std=c++17 -Wno-unused-value

HEADERS = $(shell find -name "*.h")

%.hip: %.cu $(HEADERS)
	$(HIPIFY) -o $@ $<

%.o: %.hip $(HEADERS)
	$(CXX) -c -o $@ $< $(CXX_FLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXX_FLAGS)

CU_FILES = $(shell find -name "*.cu")
HIP_FILES = $(CU_FILES:.cu=.hip)
CPP_FILES = $(shell find -name "*.cpp")
OBJS = $(HIP_FILES:.hip=.o) $(CPP_FILES:.cpp=.o)

hello: $(OBJS)
	$(CXX) -o $@ $^

clean:
	rm $(OBJS) hello

