APP_NAME=knowledgeable-prompt-tuning
VERSION_CODE=0.1.0
IMAGE_NAME=$APP_NAME:$VERSION_CODE

docker run -t -i -p 4396:80 --rm \
    -e OPENAI_API_BASE=proxy_host \
    -e OPENAI_API_KEY=sk-your_api_key \
    -e PYTHONPATH=${workspaceFolder} \
    $IMAGE_NAME
