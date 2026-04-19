from sagemaker.model import Model
import sagemaker

# Create session
sess = sagemaker.Session()

# Get Role
role = "arn:aws:iam::152982431111:role/service-role/AmazonSageMaker-ExecutionRole-20260314T005535"

# Create model
model = Model(
    image_uri = "152982431111.dkr.ecr.us-west-2.amazonaws.com/sdxl",
    model_data = "s3://s3-nail-vr/model/stable-diffusion-xl-refiner-1.0/", 
    role = role,
    sagemaker_session = sess
)

# Init Predictor
predictor = model.deploy(
    instance_type = 'ml.g5.2xlarge',
    initial_instance_count = 1
)