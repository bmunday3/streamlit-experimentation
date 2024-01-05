from modzy import EdgeClient
from modzy.edge import InputSource

client = EdgeClient('localhost', 55000)
client.connect()

img_bytes = open("test.jpg", "rb").read()
input_object = InputSource(
    key="image",
    data=img_bytes
) 

inference = client.inferences.perform_inference("avlilexzvu", "1.0.0", [input_object]) # direct mode
print(inference)