import urllib.request
import json
import os

def download_file(node_id, target_filename, dest_path):
    url = f"https://api.osf.io/v2/nodes/{node_id}/files/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        providers = json.loads(response.read().decode())['data']
        
    osfstorage = next(p for p in providers if p['attributes']['name'] == 'osfstorage')
    files_url = osfstorage['relationships']['files']['links']['related']['href']
    
    def find_and_download(url):
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            for item in data['data']:
                attrs = item['attributes']
                kind = attrs['kind']
                name = attrs['name']
                
                if kind == 'file' and name.lower() == target_filename.lower():
                    download_url = item['links']['download']
                    print(f"Downloading {name} from {download_url} to {dest_path}")
                    urllib.request.urlretrieve(download_url, dest_path)
                    print("Download complete.")
                    return True
                elif kind == 'folder':
                    folder_url = item['relationships']['files']['links']['related']['href']
                    if find_and_download(folder_url):
                        return True
        return False

    if not find_and_download(files_url):
        print(f"File {target_filename} not found in OSF node.")

if __name__ == '__main__':
    download_file("6594x", "sub_name_stim.mat", "dataset/sub_name_stim.mat")
