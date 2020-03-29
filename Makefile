install:
		pip install .; cd ./spiketrainn; make

black:
		black -S ./spiketrainn ./examples
