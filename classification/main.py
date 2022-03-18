import torch
import os
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
from dataset import *
from train_val import *
from utils import *
from model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--GPU_NUM', type=int, default=0)
parser.add_argument('--model_name', type=str, default='vgg16')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=int, default=0.0001)
parser.add_argument('--weight_decay', type=int, default=0.05)
parser.add_argument('--seed_num', type=int, default=82)
opt = parser.parse_args()

file_name = opt.model_name

### Device Setting ###
device = torch.device(f'cuda:{opt.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
print('\t' * 3 + 'Current cuda device : ', torch.cuda.current_device())
print('\t' * 3 + f'{torch.cuda.get_device_name(opt.GPU_NUM)}')
print('\t' * 3 + opt.model_name)

### Directories ###
train_dir = os.path.join(opt.root + 'dataset/train/')
test_dir = os.path.join(opt.root + 'dataset/test/')
train_gt_dir = os.path.join(opt.root + 'dataset/train_gt/')
test_gt_dir = os.path.join(opt.root + 'dataset/test_gt/')
label_dir = os.path.join(opt.root + 'label/label_512.csv')
test_label_dir = os.path.join(opt.root + 'label/label_test_512.csv')
## #Make Directories ##

os.makedirs(f'/hdd1/jiho_data/powerline/VM_dataset/model/{file_name}', exist_ok=True)
os.makedirs(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/train/{file_name}', exist_ok=True)
os.makedirs(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/train_gt/{file_name}_gt', exist_ok=True)
os.makedirs(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/test/{file_name}', exist_ok=True)
os.makedirs(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/test_gt/{file_name}_gt', exist_ok=True)
os.makedirs('./record/', exist_ok=True)

### Import model ###
model = initialize_model('vgg16', feature_extract=True, use_pretrained=True) #opt.model_name 'vgg16'
model.apply(weights_init_normal)

features = list(model.features.children())
features.insert(29, dilate_attn(512))

# feature_dict = {3:64, 9:128, 17:256, 25:512, 33:512} # apply attention module in each block
# for i in feature_dict: #3,7,13,19,25
#     features.insert(i, dilate_attn(feature_dict[i]))

model.features = nn.Sequential(*features)
get_feature(model)

pretrained_dict = models.vgg16(pretrained=True).state_dict()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if (k in model_dict) and
                       (model_dict[k].shape == pretrained_dict[k].shape)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

### Import dataset ###
train_set = pl_dataset(train_dir, label_dir)
train_dataset, val_dataset = random_split(train_set, [300,50], generator=torch.Generator().manual_seed(opt.seed_num))
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

### Define loss function, optimizer ###.
loss_func = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)

### Train ##
record = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc','val_acc'])
evaluate_record = pd.DataFrame(columns=['iter', 'TP_R', 'FN_R', 'TP_P', 'FP_P', 'Recall', 'Precision', 'FDR'])
for epoch in range(0, opt.num_epochs):
    model.to(device)
    train_loss_, train_accuracy_ = train(model, train_loader, optimizer, loss_func, device)
    val_loss_, val_accuracy_ = validation(model, val_loader, loss_func, device)
    print('===' * 26)
    print('[epoch : {}]'.format(epoch))
    print('Train Loss : {:.5f}, Accuracy : {:.3f}% \t lr : {:.7f}'.format(train_loss_, train_accuracy_, opt.lr))
    print('Validation Loss : {:.5f}, Accuracy : {:.3f}%'.format(val_loss_, val_accuracy_))
    print('===' * 26)
    if epoch % 1 == 0:
        record.loc[(len(train_loader)*epoch)] = [epoch, train_loss_, train_accuracy_, val_loss_, val_accuracy_]
    record.to_csv(f'./record/{file_name}.csv', header=True, index=False)
    torch.save(model.state_dict(), f'/hdd1/jiho_data/powerline/VM_dataset/model/{file_name}/{epoch}.pth')
    opt.lr = opt.lr - (0.05 * opt.lr)

test_data = patch_dataset(test_dir, test_label_dir , test_gt_dir) # Test image
test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, drop_last=False)
for num, (data, target, gt) in enumerate(test_dataloader):
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        output = model(data)
        pred = (output > 0.5).float()
        final_mask = mask_extract()
        otsu_mask_, mask_, ori_data = visualize_images(final_mask, data, pred)

        for i in range(len(mask_)):
            gt_img = gt[i].detach()
            gt_img = gt_img.numpy()
            mask_img = mask_[i]
            mask_img = np.squeeze(mask_img)
            if pred[i][0] == 1:
                cv2.imwrite(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/test/{file_name}/{num}_{i + 1}.bmp', mask_img)
                cv2.imwrite(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/test_gt/{file_name}_gt/{num}_{i + 1}.bmp', gt_img)

print('train mask extract')
train_data = patch_dataset(train_dir, label_dir , train_gt_dir) # Train image
train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, drop_last=False)
for num, (data, target, gt) in enumerate(train_dataloader):
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        output = model(data)
        pred = (output > 0.5).float()
        final_mask = mask_extract()
        otsu_mask_, mask_, ori_data = visualize_images(final_mask, data, pred)

        for i in range(len(mask_)):
            gt_img = gt[i].detach()
            gt_img = gt_img.numpy()
            mask_img = mask_[i]
            mask_img = np.squeeze(mask_img)
            #print(pred[i][0])
            if pred[i][0] == 1:
                cv2.imwrite(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/train/{file_name}/{num}_{i + 1}.bmp', mask_img)
                cv2.imwrite(f'/hdd1/jiho_data/powerline/VM_dataset/vgg16/train_gt/{file_name}_gt/{num}_{i + 1}.bmp', gt_img)

print('\n' + '---' * 10 + 'End' + '---' * 10)

all = [var for var in globals() if var[0] != '_']
for var in all:
    del globals()[var]


