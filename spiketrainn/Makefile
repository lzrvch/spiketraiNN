all: distance_binding.so

distance_binding.so: distance.o distance_binding.o
	g++ -O3 -shared distance_binding.o distance.o -lpthread -lm -o $@

distance_binding.o: distance_binding.cpp
	g++ -c -Wno-cpp -fPIC -O3 `python3.7-config --includes` -I `python3.7-config --prefix`/lib/python3.7/site-packages/numpy/core/include/ distance_binding.cpp -o $@

distance_binding.cpp: distance_binding.pyx
	cython --cplus distance_binding.pyx -o $@

distance.o: distance.cpp distance.h
	g++ distance.cpp -O3 -std=c++11 -c -fPIC -o $@

clean:
	rm distance_binding.cpp distance.o distance_binding.o distance_binding.so
