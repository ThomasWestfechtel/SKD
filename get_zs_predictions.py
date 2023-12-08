import numpy as np
import torch
import clip
import argparse
from data_list import ImageList
import pre_process as prep
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Token finder')
parser.add_argument('--dset', type=str, default='office-home', choices=['visda', 'office-home', 'domain-net'], help="The dataset used")
parser.add_argument('--template', type=str, default='ImageNetSm', choices=['simple', 'CIFAR' ,'ImageNet', 'ImageNetSm'], help="The template for text encoder")
parser.add_argument('--model', type=str, default='res', choices=['res', 'ViT'], help="Model to use")

args = parser.parse_args()

if(args.dset == "visda"):
    domains = ["source","target"]
if(args.dset == "office-home"):
    domains = ["Art","Clipart","Product","RealWorld"]
if(args.dset == "domain-net"):
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

device = "cuda"
if(args.model == 'ViT'):
    model, preprocess = clip.load("ViT-B/16", device=device)
elif(args.dset == "office-home"):
    model, preprocess = clip.load("RN50", device=device)
else:
    model, preprocess = clip.load("RN101", device=device)

templates = []
classes = []

f = open("ZS_Predictions/templates_" + args.template + ".txt", "r")
for x in f:
	s = x.find("'")
	e = x.rfind("'")
	if(s>=0):
		templates.append(x[s+1:e])
f.close()

if(args.dset == 'visda'):
    dset_str = "vis"
    f = open("ZS_Predictions/vis_classes.txt")
elif(args.dset == 'office-home'):
    dset_str = "oh"
    f = open("ZS_Predictions/oh_classes.txt")
elif(args.dset == 'domain-net'):
    dset_str = "dn"
    f = open("ZS_Predictions/dn_classes.txt")

for x in f:
	s = x.find("'")
	e = x.rfind("'")
	if(s>=0):
		classes.append(x[s+1:e])
f.close

zeroshot_weights = zeroshot_classifier(classes, templates)

def image_classification(loader, model, zs_weight):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            feats = model.visual(inputs)
            logit_scale = model.logit_scale.exp() / 100
            outputs = logit_scale * feats @ zs_weight
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_feats = feats.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_feats = torch.cat((all_feats, feats.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output, all_feats

for d in domains:
    prep_dict = prep.image_test()
    if (args.dset == 'visda'):
        if d == "source":
            ds = "train"
        else:
            ds = "validation"
        dset = ImageList(open("/home/tommek/VisDA/" + ds + "/image_list.txt").readlines(), transform=prep_dict)
    elif (args.dset == 'office-home'):
        dset = ImageList(open("/home/tommek/OfficeHome/" + d + "/label_c.txt").readlines(), transform=prep_dict)
    elif (args.dset == 'domain-net'):
        dset = ImageList(open("/home/tommek/DomainNet/" + d + "/label_c.txt").readlines(), transform=prep_dict)

    dset_loader = DataLoader(dset, batch_size=32, shuffle=False, num_workers=16)

    model.train(False)
    print("Starting predictions:")
    temp_acc, preds, feats = image_classification(dset_loader, model, zeroshot_weights)
    log_str = "Precision: {:.5f}".format(temp_acc)
    print(log_str)
    preds = preds.numpy()

    saveString = "Predictions_" + args.model + "_" + dset_str + "/" + d + "_" + args.template + ".npy"
    np.save(saveString, preds)