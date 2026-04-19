import sagemaker
from sagemaker.serve import ModelBuilder
#from sagemaker.model import Model

# Create session
sess = sagemaker.Session()

# Get Role
role = "arn:aws:iam::152982431111:role/service-role/AmazonSageMaker-ExecutionRole-20260314T005535"
image_uri = "152982431111.dkr.ecr.us-west-2.amazonaws.com/sdxl"
model_data = "s3://s3-nail-vr/model/stable-diffusion-xl-refiner-1.0/", 

# # Create model
# model = Model(
#     image_uri = "152982431111.dkr.ecr.us-west-2.amazonaws.com/sdxl",
#     model_data = "s3://s3-nail-vr/model/stable-diffusion-xl-refiner-1.0/", 
#     role = role,
#     sagemaker_session = sess
# )

# Create model builder
model_buider = ModelBuilder(
    model = 'sdxl',
    model_path = model_data,
    image_url = image_uri,
    role = role
)

model = model_buider.build(model_name = 'sdxl_model')

endpoint = model_buider.deploy(
    endpoint_name = 'sdxl_endpoint',
    instance_type= 'ml.m5.xlarge',
    initial_instance_count= 1
)

# # Init Predictor
# predictor = model.deploy(
#     instance_type = 'ml.g5.2xlarge',
#     initial_instance_count = 1
# )