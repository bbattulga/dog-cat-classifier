from google.cloud import storage
import os

credential_path = '/Users/battulga/env/gcp/gcp-client.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# If you don't specify credentials when constructing the client, the
# client library will look for credentials in the environment.
storage_client = storage.Client()

# Make an authenticated API request
buckets = list(storage_client.list_buckets())
print(buckets)
