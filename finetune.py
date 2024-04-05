import torch
import pandas as pd
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
import pdb
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
import multiprocessing
import torchvision
import warnings
from PIL import Image
import numpy as np
import wandb
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

class CustomResNet(nn.Module):
    def __init__(self, original_model):
        super(CustomResNet, self).__init__()
        self.features = nn.Sequential(
            *list(original_model.children())[:-1]
        )
        num_ftrs = original_model.fc.in_features
        self.new_fc_sequence = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.LayerNorm(512),  
            nn.ReLU(),
            nn.Linear(512, 100)
        )
        self.output_act = nn.LogSoftmax(dim=-1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.new_fc_sequence(x)
        x = self.output_act(x)
        return x

    def _initialize_weights(self):
        for m in self.new_fc_sequence:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CelebDataset(Dataset):
    def __init__(self,img_dir,label_df, label_map, transform = None):
        self.img_dir = img_dir
        self.labels= label_df
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx,1])
        label_name = self.labels.iloc[idx,2]
        label_num = self.label_map[label_name]
        img = Image.open(img_path)
        if self.transform:
          img = self.transform(img)

        sample = {'images': img, 'labels': label_num, 'id': self.labels.iloc[idx,1]}
        return sample

    def collate_fn(self,batch):
        batch_data = {'images': [], 'labels': [], 'ids': []}

        for b in batch:
            batch_data['images'].append(b['images'])
            batch_data['labels'].append(torch.tensor(b['labels'], dtype=torch.long))
            batch_data['ids'].append(b['id'])
        batch_data['images'] = torch.stack(batch_data['images'], dim=0)
        batch_data['labels'] = torch.stack(batch_data['labels'], dim=0)

        return batch_data

class CelebDatasetTest(Dataset):
    def __init__(self,img_dir, transform = None):
        self.img_dir = img_dir
        self.transform = transform

        entries = os.listdir(img_dir)
        self.files = [entry for entry in entries if os.path.isfile(os.path.join(self.img_dir, entry))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        img_path = self.files[idx]
        img = Image.open(os.path.join(self.img_dir,img_path))
        id = img_path.split('.')[0]
        if self.transform:
          img = self.transform(img)

        sample = {'images': img, 'id': id}
        return sample

    def collate_fn(self,batch):
        batch_data = {'images': [], 'ids': []}

        for b in batch:
            batch_data['images'].append(b['images'])
            batch_data['ids'].append(b['id'])
        batch_data['images'] = torch.stack(batch_data['images'], dim=0)


        return batch_data

def preprocess_img(directory):
    """
    Preprocess images by iterating through all files in the specified directory (and its subdirectories),
    checking their format, attempting to load them, and deleting files that are either in unsupported formats
    or cause errors during loading.

    Parameters:
    - directory: The root directory to start processing from.
    """
    # Define supported formats
    supported_formats = ['.jpeg', '.png']
    num_deleted = 0  # Counter for deleted files

    for root, dirs, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(root, file)  # Construct the full path of the image
            _, ext = os.path.splitext(img_path)  # Extract file extension

            # # Check if the file extension is not in the supported formats
            if ext.lower() not in supported_formats:
                print(f"Unsupported image format for file: {img_path}. Deleting file.")
                os.remove(img_path)  # Delete the unsupported file
                num_deleted += 1
                continue  # Skip the rest of the loop

            # Attempt to load the image
            try:
                img = read_image(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Deleting file.")
                os.remove(img_path)  # Delete the problematic file
                num_deleted += 1

    print(f"Preprocessing complete. Total files deleted: {num_deleted}")



def preprocess_df(df, img_dir):
    # Create a full path column if the paths are relative
    df['full_image_path'] = df['File Name'].apply(lambda x: os.path.join(img_dir, x))
    df['file_exists'] = df['full_image_path'].apply(os.path.exists)
    filtered_df = df[df['file_exists']].copy()
    filtered_df.drop(columns=['full_image_path', 'file_exists'], inplace=True)

    return filtered_df

def accuracy(output, label):
    pred = torch.argmax(output, dim =-1)
    correct = torch.sum((pred == label))
    return correct/output.shape[0]


def test(model_dir):
    warnings.filterwarnings("ignore", message="libpng warning")
    batch_size = 128
    category = pd.read_csv('category.csv')
    label_map = pd.Series(category.Category.values,index=category['Unnamed: 0']).to_dict()
    rev_label_map = {name:ind for ind, name in label_map.items()}
    # label_map = {i: name for i, name in enumerate(category)}
    target_width = 224
    data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize(target_width+1, antialias=True),# need to resize it to one more up scale since some of width and height are different
        transforms.CenterCrop(target_width),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        transforms.ToTensor(),  # makes 0~1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize(target_width+1, antialias=True),
        transforms.CenterCrop(target_width),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_cores = multiprocessing.cpu_count()-2
    
    # debugging purpose
    # num_cores = 0
    # raw_csv = pd.read_csv('train.csv')
    # raw_csv = preprocess_df(raw_csv,'train_extracted') # get rid of deleted image
    # train_df, val_df = train_test_split(raw_csv, test_size=0.2, stratify=raw_csv['Category'])
    # train_dataset = CelebDataset('train_extracted',
    #                             train_df,
    #                              rev_label_map,
    #                             data_transforms['train'])
    # train_loader = DataLoader(train_dataset, batch_size =batch_size, pin_memory= True, collate_fn = train_dataset.collate_fn,
    #                           num_workers =0)
    
    # val_dataset = CelebDataset('train_extracted',
    #                             val_df,
    #                            rev_label_map,
    #                             data_transforms['val'])
    # val_loader = DataLoader(val_dataset, batch_size =batch_size, pin_memory= True, collate_fn = val_dataset.collate_fn,
    #                           num_workers =num_cores)
    
    test_dataset = CelebDatasetTest('test_extracted',
                                data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size =batch_size, pin_memory= True, collate_fn = test_dataset.collate_fn,
                              num_workers =num_cores)
    model = CustomResNet(torchvision.models.resnet101())
    model.to(device)

    state = torch.load(model_dir)
    model.load_state_dict(state)
    model.eval()
    
    label_list = []
    id_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            images = data['images'].to(device)
            ids = data['ids']
            outputs = model(images)
            labels = torch.argmax(outputs,dim=-1)
            
            names = [label_map[label.item()] for label in labels]
            # pdb.set_trace()
            # x=torch.sum(data['labels'].to(device)==labels.to(device))
            # acc = x/labels.shape[0]
            label_list.extend(names)
            id_list.extend(ids)

    celebrity_names = category

    
    df_pred = pd.DataFrame({
    'Id': id_list,
    'Category': label_list,
    })
    df_pred.sort_values(by='Id', inplace=True)
    df = pd.read_csv('sample_submission.csv')
    for idx, row in df_pred.iterrows():
        match_index = df[df['Id'] == int(row['Id'])].index
        if not match_index.empty:
            df.loc[match_index, 'Category'] = row['Category']
    csv_file_path = 'prediction.csv'
    df.to_csv(csv_file_path, index=False)




def finetune():
    if wandb_on:
        wandb.init(group='rest101')
        config = wandb.config
        batch_size = config.batch
        learning_rate = config.learning_rate
        reg = config.reg
        patients = 3
    else:
        learning_rate = 1e-5
        patients=5
        reg = 0.001
        batch_size = 32

    warnings.filterwarnings("ignore", message="libpng warning")

    raw_csv = pd.read_csv('train.csv')
    raw_csv = preprocess_df(raw_csv,'train_extracted') # get rid of deleted image
    category = pd.read_csv('category.csv')
    label_map = pd.Series(category['Unnamed: 0'].values, index=category.Category).to_dict()
    train_df, val_df = train_test_split(raw_csv, test_size=0.2, stratify=raw_csv['Category'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    target_width=225
    data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize(target_width+1, antialias=True),# need to resize it to one more up scale since some of width and height are different
        transforms.CenterCrop(target_width),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        transforms.ToTensor(),  # makes 0~1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize(target_width+1, antialias=True),
        transforms.CenterCrop(target_width),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

    train_dataset = CelebDataset('train_extracted',
                                train_df,
                                 label_map,
                                data_transforms['train'])
    val_dataset = CelebDataset('train_extracted',
                                val_df,
                               label_map,
                                data_transforms['val'])

    # # Define the size of the subset for training and validation
    # train_subset_size = int(len(train_df)*0.5)
    # val_subset_size = int(len(val_df)*0.5)

    # train_indices = np.random.choice(len(train_df), train_subset_size, replace=False)
    # val_indices = np.random.choice(len(val_df), val_subset_size, replace=False)

    # train_subset = Subset(train_dataset, train_indices)
    # val_subset = Subset(val_dataset, val_indices)



    num_cores = multiprocessing.cpu_count()-2
    # num_cores = 0
    train_loader = DataLoader(train_dataset, batch_size =batch_size, pin_memory= True, collate_fn = train_dataset.collate_fn,
                              num_workers =num_cores)
    val_loader = DataLoader(val_dataset, batch_size =batch_size, pin_memory= True, collate_fn = val_dataset.collate_fn,
                              num_workers =num_cores)

    model = CustomResNet(torchvision.models.resnet101())
    model.to(device)
    num_epochs = 150
    print_interval = 10


    best_val_acc = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=reg)
    criterion = nn.NLLLoss()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=len(train_loader), num_training_steps=int(1.1 * num_epochs * len(train_loader))
    )
    for epoch in range(num_epochs+1):
        total_loss = 0
        total_correct = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # train
        model.train()
        for data in train_loader:
            images = data['images'].to(device)
            labels = data['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            if torch.isnan(train_loss):
                break

            train_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(),1)
            optimizer.step()
            scheduler.step()
            
            train_acc = accuracy(outputs.detach(), labels)
            train_losses.append(train_loss.detach())
            train_accs.append(train_acc.detach())
        epoch_train_acc = sum(train_accs)/ len(train_accs)
        epoch_train_loss = sum(train_losses)/ len(train_losses)

        if epoch % print_interval == 0:
            print("Epoch {}, Train_Loss={}, Train_Accuracy={}".format(epoch, round(epoch_train_loss.item(),4),round(epoch_train_acc.item()*100,2)))

        # validation
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                images = data['images'].to(device)
                labels = data['labels'].to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_acc = accuracy(outputs.detach(), labels)
                val_losses.append(val_loss.detach())
                val_accs.append(val_acc.detach())
        epoch_val_acc = sum(val_accs)/ len(val_accs)
        epoch_val_loss = sum(val_losses)/ len(val_losses)

        if best_val_acc < epoch_val_acc:
            bset_val_acc = epoch_val_acc
            wandb.log({'best_val_acc':bset_val_acc})
            torch.save(model.state_dict(), 'model_best_val_acc.pth')
            early_stop_count=0
        else:
            early_stop_count+=1

        if early_stop_count > patients:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

        if epoch % print_interval == 0:
            print("Epoch {}, Val_Loss={}, Val_Accuracy={}".format(epoch, round(epoch_val_loss.item(),4),round(epoch_val_acc.item()*100,2)))

        #wandb loging
        if wandb_on:
            wandb.log({'epoch':epoch, 'train_loss':epoch_train_loss, 'train_acc':epoch_train_acc, 'val_loss':epoch_val_loss, 'val_acc':epoch_val_acc})


    return
if __name__ == '__main__':
    # preprocess_img('train')
    # remove warnings
    # wandb_on = True
    # if wandb_on:
    #     sweep_config = dict()
    #     sweep_config['method'] = 'grid'
    #     sweep_config['metric'] = {'name': 'val_acc', 'goal': 'maximize'}
    #     sweep_config['parameters'] = {'learning_rate': {'values' : [0.001]}, 'reg' : {'values':[0.001]},
    #                                   'batch':{'values':[128]}}

    #     sweep_id = wandb.sweep(sweep_config, project = 'Purdue Face Recognition')
    #     wandb.agent(sweep_id, finetune)
    # else:
    #     finetune()
    test('model_best_val_acc.pth')















