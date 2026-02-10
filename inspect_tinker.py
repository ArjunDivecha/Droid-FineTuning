import tinker
import inspect

print("\ntinker.ServiceClient.create_sampling_client signature:")
try:
    print(inspect.signature(tinker.ServiceClient.create_sampling_client))
except Exception as e:
    print(f"Could not get signature: {e}")

print("\ntinker.ServiceClient.create_sampling_client doc:")
print(tinker.ServiceClient.create_sampling_client.__doc__)

