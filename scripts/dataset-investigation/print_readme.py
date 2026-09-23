import urllib.request
import json

def get_readme(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        
        for item in data['data']:
            attrs = item['attributes']
            kind = attrs['kind']
            name = attrs['name']
            
            if kind == 'file' and name == 'readme.txt':
                download_url = item['links']['download']
                req = urllib.request.Request(download_url)
                with urllib.request.urlopen(req) as r:
                    print("--- README CONTENT ---")
                    print(r.read().decode('utf-8'))
                return True
            elif kind == 'folder':
                folder_url = item['relationships']['files']['links']['related']['href']
                if get_readme(folder_url):
                    return True
                    
        if 'links' in data and 'next' in data['links'] and data['links']['next']:
            return get_readme(data['links']['next'])
            
    return False

def fetch_osf_files(node_id):
    url = f"https://api.osf.io/v2/nodes/{node_id}/files/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        providers = json.loads(response.read().decode())['data']
        
    osfstorage = next(p for p in providers if p['attributes']['name'] == 'osfstorage')
    files_url = osfstorage['relationships']['files']['links']['related']['href']
    get_readme(files_url)

if __name__ == "__main__":
    fetch_osf_files("6594x")
