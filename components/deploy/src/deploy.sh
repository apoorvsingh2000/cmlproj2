# loop through all parameters
while [ "$1" != "" ]; do
    case $1 in
      "--model-path")
        shift
        MODEL_PATH="$1"
        shift
        ;;
      "--model-name")
        shift
        MODEL_NAME="$1"
        shift
        ;;
      "--model-region")
        shift
        MODEL_REGION="$1"
        shift
        ;;
      "--model-version")
        shift
        MODEL_VERSION="$1"
        shift
        ;;
      "--model-runtime-version")
        shift
        RUNTIME_VERSION="$1"
        shift
        ;;
      "--model-prediction-class")
        shift
        MODEL_PREDICTION_CLASS="$1"
        shift
        ;;
      "--model-python-version")
        shift
        MODEL_PYTHON_VERSION="$1"
        shift
        ;;
      "--model-package-uris")
        shift
        MODEL_PACKAGE_URIS="$1"
        shift
        ;;
      *)
        shift
        ;;
   esac
done

# echo inputs
echo MODEL_PATH               = "${MODEL_PATH}"
echo MODEL_NAME               = "${MODEL_NAME}"
echo MODEL_REGION             = "${MODEL_REGION}"
echo MODEL_VERSION            = "${MODEL_VERSION}"
echo RUNTIME_VERSION          = "${RUNTIME_VERSION}"
echo MODEL_PREDICTION_CLASS   = "${MODEL_PREDICTION_CLASS}"
echo MODEL_PYTHON_VERSION     = "${MODEL_PYTHON_VERSION}"
echo MODEL_PACKAGE_URIS       = "${MODEL_PACKAGE_URIS}"

# Check if model exists
modelname=$(gcloud vertex ai models list --region=${MODEL_REGION} --filter="displayName=${MODEL_NAME}" --format="value(displayName)")

if [ -z "$modelname" ]; then
   echo "Creating model $MODEL_NAME in region $MODEL_REGION"
   gcloud vertex ai models upload \
     --region=${MODEL_REGION} \
     --display-name=${MODEL_NAME} \
     --artifact-uri=${MODEL_PATH} \
     --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.${RUNTIME_VERSION}:latest
else
   echo "Model $MODEL_NAME already exists in region $MODEL_REGION"
fi

# Deploy a new model version (optional for more advanced deployments)
# Here we assume a single version exists.
