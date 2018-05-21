
#!/bin/bash

xhost +local:docker

docker run --privileged --runtime=nvidia -it --rm \
  --env DISPLAY=$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --volume $PWD:/code --name=face_rec \
  narenm/opencv:gpu bash

xhost -local:docker