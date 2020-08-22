# Docker Containers

This document provides a quick start for users interested in using or deploying this code using docker containers. Docker containers provide a portable and controlled environment to assist in consistency and reliability in deployments.

## The Containers

We offer multiple containers for specific use cases. This allows us to ensure that the containers are only as big as required for specific functionality. Certain containers build off of others. Example code below is intended to run in the home directory of this project / repository.

## Base

This image contains the bare minimum for deploying the package.

Build the image:
```
docker build \
  --build-arg http_proxy=${http_proxy} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg no_proxy=${no_proxy} \
  -t base \
  -f docker/base/Dockerfile .
```

Use the image:
```
docker run -dti -v $(pwd):/root basdocker ps
docker attach <container_name>
```

## Testing

This image contains the package and any utilities required for testing.

Build the image:
```
docker build \
  --build-arg http_proxy=${http_proxy} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg no_proxy=${no_proxy} \
  -t tests \
  -f docker/tests/Dockerfile .
```

Use the image:
```
docker run -dti -v --name silly_name_test $(pwd):/root tests
docker attach silly_name_test
```

## Cleaning Up

To stop all containers:
```
docker container stop $(docker container ls -aq)
```

To delete all containers:
```
docker container rm $(docker container ls -aq)
```

To remove dangling and unused containers:
```
docker image prune -a
```
