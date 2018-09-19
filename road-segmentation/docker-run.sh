#!/usr/bin/env bash

nvidia-docker run \
	-v /home/sheetal/Movehack/satellite_imagery/RoadDetector/albu-solution/src:/opt/app/src/ \
	-v /disk1/satellite/SpaceNet_Roads_Sample:/data:ro \
	-v /home/sheetal/Movehack/satellite_imagery/RoadDetector/albu-solution/output:/wdata \
	--rm -ti --ipc=host \
	satellite/albu
