TYPE = float
CXXFLAGS = -Wall

ifeq ($(TYPE), double)
    CXXFLAGS += -DUSE_DOUBLE
endif

all: task1

main: task1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f task1