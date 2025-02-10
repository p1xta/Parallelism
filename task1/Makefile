TYPE = float
CXXFLAGS = -Wall

ifeq ($(TYPE), double)
    CXXFLAGS += -DDOUBLE
endif

all: task1

task1: task1.cpp
	g++ $(CXXFLAGS) -o $@ $<

clean:
	rm -f task1