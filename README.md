# KSD
PyTorch Code for KSD -Combining inherent knowledge of vision-language models with unsupervised domain adaptation through self-knowledge distillation
https://arxiv.org/abs/2312.04066

### Method overview
Unsupervised domain adaptation (UDA) tries to overcome the tedious work of labeling data by leveraging a labeled source dataset and transferring its knowledge to a similar but different target dataset. On the other hand, current vision-language models exhibit astonishing zero-shot prediction capabilities. In this work, we combine knowledge gained through UDA with the inherent knowledge of vision-language models.
In a first step, we generate the zero-shot predictions of the source and target dataset using the vision-language model. Since zero-shot predictions usually exhibit a large entropy, meaning that the class probabilities are rather evenly distributed, we first adjust the distribution to accentuate the winning probabilities. This is done using both source and target data to keep the relative confidence between source and target data. We then employ a conventional DA method, to gain the knowledge from the source dataset, in combination with self-knowledge distillation, to maintain the inherent knowledge of the vision-language model. 
We further combine our method with a gradual source domain expansion strategy (GSDE) and show that this strategy can also benefit by including zero-shot predictions. 
We conduct experiments and ablation studies on three benchmarks (OfficeHome, VisDA, and DomainNet) and outperform state-of-the-art methods. We further show in ablation studies the contributions of different parts of our algorithm.

### Usage
For ViT backbone:  
To train the network on OfficeHome: 	bash train_oh_ViT.sh 0  
where 0 is the task id [0...12]  
To train the network on VisDA: 	bash train_vis_ViT.sh  

For ResNet backbone:  
To train the network on OfficeHome: 	bash train_oh_res.sh 0  
where 0 is the task id [0...12]  
To train the network on VisDA: 	bash train_vis_res.sh  

You have to change s_dset_path and t_dset_path to the location of the dataset.  

label_c.txt consists of the full path to the image file and the class id  
Example from art of OfficeHome:  

/home/xxx/OfficeHome/Art/Alarm_Clock/00001.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00002.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00003.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00004.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00005.jpg 0  
...  
/home/xxx/OfficeHome/Art/Backpack/00001.jpg 1  
/home/xxx/OfficeHome/Art/Backpack/00002.jpg 1  
/home/xxx/OfficeHome/Art/Backpack/00003.jpg 1  
/home/xxx/OfficeHome/Art/Backpack/00004.jpg 1  

get_zs_predictions.py is for getting the zero predicions for the datasets

### Citation
If you use KSD code please cite:
```text
@article{GSDE,
  title={Combining inherent knowledge of vision-language models with unsupervised domain adaptation through self-knowledge distillation},
  author={Westfechtel, Thomas and Zhang, Dexuan and Harada, Tatsuya},
  journal={arXiv preprint arXiv:2312.04066},
  year={2023}
}
```
