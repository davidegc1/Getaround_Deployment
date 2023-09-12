# open terminal at Dashboard folder
docker build . -t getaround_dashboard_image
docker run -it -v "$(pwd):/home/app" -p 4002:4002 -e PORT=4002 getaround_dashboard_image