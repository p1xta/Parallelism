TYPE = float
CXXFLAGS = -Wall

ifeq ($(TYPE), double)
    CXXFLAGS += -DDOUBLE
endif

all: task1

main: task1.cpp
	$(CXXFLAGS) -o $@ $<

clean:
	rm -f task1