IMAGE_TAG=${1:-"$(whoami)/deepfakes"}
DOCKERFILE_DIR=${2:-"./docker"}

# Human-readable parameters for Docker build
SAVE_IMAGE_UNDER_GIVEN_TAG="--tag $IMAGE_TAG"
BUILD_FROM_DOCKERFILE_IN_GIVEN_DIR=$DOCKERFILE_DIR

docker build \
        $SAVE_IMAGE_UNDER_GIVEN_TAG \
        $BUILD_FROM_DOCKERFILE_IN_GIVEN_DIR