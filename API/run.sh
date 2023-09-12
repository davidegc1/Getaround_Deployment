# open new terminal at API folder
docker build . -t getaround_api_image
docker run -it -v "$(pwd):/home/app" -p 4001:4001 -e PORT=4001 getaround_api_image