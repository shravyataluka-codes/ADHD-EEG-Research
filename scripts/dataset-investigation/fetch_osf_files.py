import urllib.request
import json
import urllib.parse

def list_files(url, prefix=""):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        
        for item in data['data']:
            attrs = item['attributes']
            kind = attrs['kind']
            name = attrs['name']
            size = attrs.get('size', 0)
            
            if kind == 'file':
                print(f"{prefix}- {name} (File, Size: {size} bytes, ~{size/(1024*1024):.2f} MB)")
            elif kind == 'folder':
                print(f"{prefix}- {name}/ (Folder)")
                folder_url = item['relationships']['files']['links']['related']['href']
                list_files(folder_url, prefix + "  ")
                
        if 'links' in data and 'next' in data['links'] and data['links']['next']:
            list_files(data['links']['next'], prefix)

def fetch_osf_files(node_id):
    url = f"https://api.osf.io/v2/nodes/{node_id}/files/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        providers = json.loads(response.read().decode())['data']
        
    osfstorage = next(p for p in providers if p['attributes']['name'] == 'osfstorage')
    files_url = osfstorage['relationships']['files']['links']['related']['href']
    
    print(f"Files in OSF Node {node_id}:")
    list_files(files_url)

if __name__ == "__main__":
    fetch_osf_files("6594x")
