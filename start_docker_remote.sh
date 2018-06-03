
#!/bin/bash

XAUTH=$HOME/.Xauthority

docker run --tty --privileged --runtime=nvidia -it --rm \
  --network=host \
  --env DISPLAY=$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  --volume /etc/machine-id:/etc/machine-id:ro \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --volume $XAUTH:/root/.Xauthority \
  --volume $PWD:/code --name=face_rec \
  -p 8888:8888 \
  narenm/opencv:gpu bash

