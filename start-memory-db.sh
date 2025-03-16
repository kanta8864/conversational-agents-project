#!/bin/bash

CONTAINER_NAME="binge_buddy_mongo"

# Load environment variables from .env (if it exists)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found! Please create one from .env.example."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed! Please install Docker and try again."
    exit 1
fi

if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "MongoDB container '$CONTAINER_NAME' is already running."
    exit 0
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "MongoDB container exists but is stopped. Restarting..."
    docker start $CONTAINER_NAME
else
    echo "Starting MongoDB container..."
    docker run -d --name $CONTAINER_NAME -p $MONGO_PORT:27017 \
        -e MONGO_INITDB_ROOT_USERNAME=$MONGO_USER \
        -e MONGO_INITDB_ROOT_PASSWORD=$MONGO_PASS \
        mongo
fi

echo "MongoDB is running on port $MONGO_PORT."
trap "docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME; exit 0" SIGINT

echo "Press Ctrl+C to stop and remove the container."
while true; do sleep 1; done
