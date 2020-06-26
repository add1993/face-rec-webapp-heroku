import io
import os
import json
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
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
from skimage import io

class MyCustomDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, index):
        img_path,label = self.dataset[index]# Some data read from a file or image
        #data = Image.open(img_path)
        data = io.imread(img_path)
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
workers = 4 

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
            ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(db_id, image_bytes):
    checkpoint_path, checkpoint_file, label_dict = get_saved_model(db_id)
    net = InceptionResnetV1(
                classify=True,
                num_classes=None
    )
    model = net
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
         checkpoint = torch.load(checkpoint_file)
         model.load_state_dict(checkpoint['net'])
         start_epoch = checkpoint['epoch']

    model.eval()
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    print(predicted_idx)
    return predicted_idx, label_dict[predicted_idx]

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
    (cwd + '/test_images_aligned/1.png', 0), 
    (cwd + '/test_images_aligned/2.png', 1), 
    (cwd + '/test_images_aligned/3.png', 2), 
    ]
    return (dataset,3)

def update_model(db_id, path):
    pass

def get_saved_model(db_id):
    return (None, None)

def train_model(db_id):
    start_epoch = 0
    batch_size = 32
    epochs = 5
    workers = 4
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
    checkpoint_path, checkpoint_file = get_saved_model(db_id)
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

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/train', methods=['GET', 'POST'])
def train():
    id = 2
    if request.method == 'POST':
        id = request.form.get('id')
        print(id)
        train_model(id)
    return jsonify({'id' : id})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        db_id = request.files['id']
        img_bytes = file.read()
        class_id, class_name = get_prediction(db_id, image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run()

