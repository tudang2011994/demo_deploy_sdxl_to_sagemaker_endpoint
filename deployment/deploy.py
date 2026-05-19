import sagemaker
from sagemaker.serve import ModelBuilder
from sagemaker.model import Model

# Create session
sess = sagemaker.Session()

# Get Role
role = "arn:aws:iam::152982431111:role/service-role/AmazonSageMaker-ExecutionRole-20260314T005535"
image_uri = "152982431111.dkr.ecr.us-west-1.amazonaws.com/sdxl"
model_data = "s3://s3-nail-vr/model/sdxl.tar.gz" 

# Create model
model = Model(
    image_uri = image_uri,
    model_data = model_data, 
    role = role,
    sagemaker_session = sess
)

# Init Predictor
print("🚀 Starting deployment...")
try:
    predictor = model.deploy(
        instance_type="ml.g5.2xlarge",
        initial_instance_count=1,
        wait=False
    )
    print(f"✅ Endpoint ready: {predictor.endpoint_name}")
except Exception as e:
    print(f"❌ Deploy failed: {e}")