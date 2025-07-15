# make build-community_integration
# make run-community_integration
# make clean-community_integration

build-%:
	docker build --network="host" -t $* -f Dockerfile.$* .

# Run the Docker container
run-%:
	docker run --shm-size=16g \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--runtime nvidia \
	-v ${PWD}:/workspace \
	-v /home/carmelog/Projects/Docker/CommunityPhysics/TheWellMHDData:/workspace/TheWellMHDData \
	--gpus device=1 \
	--network="host" \
	--rm -it $* /bin/bash

clean-%:
	docker rmi $*

.PHONY: build-% run-% clean-%
