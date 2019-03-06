sudo sh start_docker_image.sh
cd /root/
source download_in_docker.sh

To exit the image without killing running code:

 -> Ctrl + P + Q

To get back into a running image:

 -> $ sudo docker attach paperspace_GPU0

To open more than one terminal window at the same time:

 -> $ sudo docker exec -it paperspace_GPU0 bash
