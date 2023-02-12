import hashlib

def hashName(image_file):
    h = hashlib.sha256(open(image_file, 'rb').read()).hexdigest()
    return h

