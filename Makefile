FC = gfortran
FFLAGS = -O3 -fopenmp -fPIC -ffree-line-length-none

.PHONY: all debug test clean
all: build/libhelium.so
build/libhelium.so: src/kernels.f90
	mkdir -p build
	$(FC) $(FFLAGS) -Jbuild -shared $< -o $@.tmp
	mv $@.tmp $@
debug:
	$(MAKE) build/libhelium_debug.so
build/libhelium_debug.so: src/kernels.f90
	mkdir -p build
	$(FC) -O0 -g -fopenmp -fPIC -ffree-line-length-none -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow -Jbuild -shared $< -o $@.tmp
	mv $@.tmp $@
test: all debug
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python -m pytest -q
clean:
	rm -f build/*.so build/*.mod
