DOCKER_IMAGE_NAME="$(whoami)/deepfakes"

if ! docker image inspect "$DOCKER_IMAGE_NAME" &> /dev/null; then
    ./docker/build_docker.sh $DOCKER_IMAGE_NAME ./docker
fi

# Human-readable parameters for Docker run
USE_ALL_GPUS="--gpus all"
ALLOCATE_LOTS_OF_SHARED_MEMORY="--shm-size=256g"
MAKE_CODE_ACCESSIBLE_IN_CONTAINER="--volume $(pwd):/app"
MAKE_DATA_ACCESSIBLE_IN_CONTAINER="--volume /data/:/data/"
RUN_CONTAINER_INTERACTIVELY="-it"
RUN_AS_CURRENT_USER_TO_AVOID_PERMISSION_ERRORS="--user $(id -u):$(id -g)"
DELETE_CONTAINER_ON_EXIT="--rm"
RUN_SPECIFIED_IMAGE="$DOCKER_IMAGE_NAME"

docker run \
    $USE_ALL_GPUS \
    $ALLOCATE_LOTS_OF_SHARED_MEMORY \
    $MAKE_CODE_ACCESSIBLE_IN_CONTAINER \
    $MAKE_DATA_ACCESSIBLE_IN_CONTAINER \
    $RUN_CONTAINER_INTERACTIVELY \
    $RUN_AS_CURRENT_USER_TO_AVOID_PERMISSION_ERRORS \
    $DELETE_CONTAINER_ON_EXIT \
    $RUN_SPECIFIED_IMAGE