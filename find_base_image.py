from sagemaker import image_uris

image_url = image_uris.retrieve( 
    framework='pytorch',
    region = 'us-west-2',
    version= '2.1.0',
    py_version= 'py310',
    instance_type='ml.g5.xlarge',
    image_scope= 'inference'
)

print(image_url)