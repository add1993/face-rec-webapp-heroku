import requests
import base64

resp = requests.get('http://0.0.0.0:5000/')
print(resp.status_code)
print(resp.content)

b64_ims = []
im_paths = [
      './test_images_aligned/1.png', 
	   './test_images_aligned/2.png', 
      './test_images_aligned/3.png'

]
for im_path in im_paths:
    with open(im_path, 'rb') as f:
        im_b64 = base64.encodebytes(f.read()).decode('utf-8')
        b64_ims.append(im_b64)

#payload = {"images": b64_ims, "ids" : [1,2,3,4,5,6]}
#resp = requests.post('http://127.0.0.1:5000/validate', data=payload)
#print(resp.status_code)
#print(resp.content)

#payload = {"images": b64_ims, "ids" : [1,2,3], "db_id" : 1}
#resp = requests.post('https://demo2020-app.herokuapp.com/detect', json=payload)
#print(resp.status_code)
#print(resp.content)


payload = {"images": b64_ims, "ids" : [1,2,3], "db_id" : 1}
resp = requests.post('http://0.0.0.0:5000/predict', json=payload)
print(resp.status_code)
print(resp.content)
#myobj = {'id': 1}
#resp = requests.post('https://demo2020-app.herokuapp.com/train', data=myobj)
#print(resp.status_code)
#print(resp.content)

#resp = requests.post("http://127.0.0.1:5000/predict",
#                     files={"file": open('/home/axd170033/face_recognition/dataset/images_cropped/Aaron_Peirsol/Aaron_Peirsol_0004.jpg','rb')})
#print(resp.status_code)
#print(resp.json())
