import io
import os
import json
import base64
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torchvision import models
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

class MyCustomDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, index):
        img_path,label = self.dataset[index]# Some data read from a file or image
        data = Image.open(img_path)
        data = TF.to_tensor(data)
        if self.transforms is not None:
            data = self.transforms(data)
        return (data, label)

    def __len__(self):
        return len(self.dataset) # of how many data(images?) you have

app = Flask(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('Running on device: {}'.format(device))
#dataset = datasets.ImageFolder('./dataset/images_cropped')
#dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
workers = 2 

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
            ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(db_id, images):
    checkpoint_path, checkpoint_file, label_dict = get_saved_model(db_id)
    net = InceptionResnetV1(
                classify=True,
                num_classes=len(label_dict)
    )
    model = net
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
         checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
         model.load_state_dict(checkpoint['net'])
         start_epoch = checkpoint['epoch']

    model.eval()
    pred_idx = []
    labels = []
    probs = []
    my_transforms = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    for i in range(len(images)):
        image_i = images[i]
        tensor = my_transforms(image_i).unsqueeze(0)
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prob_max = torch.max(probabilities)
        score, y_hat = outputs.max(1)
        predicted_idx = y_hat.item()
        pred_idx.append(predicted_idx)
        labels.append(label_dict[predicted_idx])
        probs.append(prob_max.detach().item())
    return pred_idx, labels, probs

def crop_images(db_id):
    mtcnn = MTCNN(
            image_size=250, margin=0, min_face_size=40,
             thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, select_largest=False,
                 device=device
    )
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    crop_transform = transforms.Compose([
             transforms.Resize(512)
    ])

    images = get_images(db_id)
    dataset = MyCustomDataset(images, crop_transform)
   
def get_dataset(db_id):
    cwd = os.getcwd()
    dataset = [
    (cwd + '/app/test_images_aligned/1.png', 0),
    (cwd + '/app/test_images_aligned/2.png', 1),
    (cwd + '/app/test_images_aligned/3.png', 2),
    ]
    return (dataset,3)

def update_model(db_id, path):
    pass

def get_saved_model(db_id):
    cwd = os.getcwd()
    label_dict = {}
    label_dict[0] = "Udit Dobhal"
    label_dict[1] = "dhruv"
    label_dict[2] = "anjit"
    label_dict[3] = "rachit"
    label_dict[4] = "Manjeet"
	label_dict[5] = "Ashish"
    return (cwd+'/app/', cwd+'/app/face_rec_test_final.pth', label_dict)

def train_model(db_id):
    start_epoch = 0
    batch_size = 32
    epochs = 5
    workers = 2
    train_transform = transforms.Compose([
             transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(p=0.5),
             np.float32,
             transforms.ToTensor(),
             fixed_image_standardization
    ])
    images, num_classes = get_dataset(db_id)
    dataset = MyCustomDataset(images, train_transform)
    train_loader = DataLoader(
                    dataset,
                    num_workers=workers,
                    batch_size=batch_size
                    )
    model = InceptionResnetV1(
                 classify=True,
                 num_classes=num_classes
            ).to(device)
    checkpoint_path, checkpoint_file, label_dict = get_saved_model(db_id)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
         checkpoint = torch.load(checkpoint_file)
         model.load_state_dict(checkpoint['net'])
         start_epoch = checkpoint['epoch']
    else:
        checkpoint_path = "./checkpoint"

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = MultiStepLR(optimizer, [60, 120, 180])
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
      'fps': training.BatchTimer(),
      'acc': training.accuracy
    }

    writer = SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=600, filename_suffix='face_rec_log_')
    writer.iteration, writer.interval = 1, 10

    checkpoint_save_name = 'face_rec_test'
    ckp_dir = checkpoint_path
    ckp_name = ''
    for epoch in range(epochs):
        training.pass_epoch(
              model, loss_fn, train_loader, optimizer, scheduler,
              batch_metrics=metrics, show_running=False, device=device,
              writer=writer
        )

        if (epoch+1) % 50 == 0:
            print('Saving..')
            state = {
               'net': model.state_dict(),
               'epoch': epoch,
               'is_final' : 0
            }
            ckp_name = checkpoint_save_name+'_'+str(epoch+1)
                       #if not os.path.isdir('checkpoint'):
            os.makedirs(ckp_dir, exist_ok=True)
            torch.save(state, ckp_dir+'/'+ckp_name+'.pth')
        writer.close()

    
    state = {
        'net': model.state_dict(),
        'epoch': epochs,
        'is_final' : 1
    }
    ckp_name = checkpoint_save_name+'_final'
    os.makedirs(ckp_dir, exist_ok=True)
    save_path = ckp_dir+'/'+ckp_name+'.pth'
    torch.save(state, save_path)
    update_model(db_id, save_path)

	
def validate_images(images):
    mtcnn = MTCNN(
            image_size=250, margin=0, min_face_size=40,
             thresholds=[0.8, 0.9, 0.9], factor=0.709, post_process=True, select_largest=False,
                 device=device
    )
    
    probability = []
    bbox = []
    idx = 0
    for image_bytes in images:
        image = Image.open(io.BytesIO(image_bytes))
        boxes, probs = mtcnn.detect(image)
        if boxes is None:
            print('No faces in img : '+str(idx))
        else:
            print(len(boxes))
            print(probs)
        probability.append(probs)
        bbox.append(boxes)
        idx = idx + 1
    return (probability, bbox)
	
@app.route('/')
def hello():
    cwd = os.getcwd()
    arr = os.listdir("/app/app")
    list = " ".join(arr)
    return cwd + ' Hello World! ' + list

@app.route('/train', methods=['GET', 'POST'])
def train():
    id = 2
    if request.method == 'POST':
        id = request.form.get('id')
        print(id)
        train_model(id)
    return jsonify({'id' : id})

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        #files = request.files.to_dict(flat=False)
        payload = request.get_json()
        images_b64 = payload['images']
        ids = payload['ids']
        
        images = []
        for im_b64 in images_b64:
            im_binary = base64.b64decode(im_b64)
            images.append(im_binary)

        probability, bbox = validate_images(images)
        output = []
        for i in range(len(probability)):
            entry = {}
            entry['id'] = ids[i]
            if bbox[i] is None:
                entry['detection'] = 'None'
                entry['prob'] = 'None'
                entry['bbox'] = 'None'
            elif len(probability[i]) > 1:
                entry['detection'] = 'MultiFace'
                entry['prob'] = probability[i].tolist()
                entry['bbox'] = bbox[i].tolist()
            else:
                if probability[i][0] < 0.85:
                    entry['detection'] = 'UnclearFace'
                else:
                    entry['detection'] = 'FaceDetected'
                entry['prob'] = probability[i].tolist()
                entry['bbox'] = bbox[i].tolist()
            output.append(entry)
        return jsonify(output)
		
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        payload = request.get_json()
        images_b64 = payload['images']
        ids = payload['ids']
        db_id = payload['db_id']
        images = []
        for im_b64 in images_b64:
            im_binary = base64.b64decode(im_b64)
            images.append(im_binary)

        probs, bbox = validate_images(images)
        
        filtered_images = []
        filtered_idxs = []
        output = []
        idx = 0
        transform_tensor_to_image = transforms.ToPILImage()
        for i in range(len(probs)):
            if probs[i] is None:
                entry = {}
                entry['id'] = ids[i]
                entry['prob'] = None
                entry['class_id'] = None
                entry['class_name'] = None
                output[idx] = entry
                idx = idx + 1
            else:
                print(bbox[i][0])
                face = extract_face(Image.open(io.BytesIO(images[i])), bbox[i][0])
                img = transform_tensor_to_image(face.cpu())
                filtered_images.append(img)
                filtered_idxs.append(ids[i])
        
        class_id, class_name, probs = get_prediction(db_id, filtered_images)
        for i in range(len(class_id)):
            entry = {}
            entry['id'] = filtered_idxs[i]
            entry['prob'] = probs[i]
            entry['class_id'] = class_id[i]
            entry['class_name'] = class_name[i]
            output.append(entry)
            idx = idx + 1
        
        return jsonify(output)

if __name__ == '__main__':
    app.run()

