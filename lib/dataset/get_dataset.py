import torchvision.transforms as transforms
from dataset.recipe1m import Recipe1M
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp
import torch

def get_datasets(args):

    if args.aug_type=='q2l':
        if args.orid_norm:
            normalize = transforms.Normalize(mean=[0, 0, 0],
                                             std=[1, 1, 1])
            # print("mean=[0, 0, 0], std=[1, 1, 1]")
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

        train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                     RandAugment(),
                                     transforms.ToTensor(),
                                     normalize]
        try:
            # for q2l_infer scripts
            if args.cutout:
                print("Using Cutout!!!")
                train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
        except Exception as e:
            Warning(e)
        train_data_transform = transforms.Compose(train_data_transform_list)

        test_data_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize])
    elif args.aug_type == 'ret':
        resize=256
        im_size=224
        train_transform_list = [transforms.Resize((resize))]
        train_transform_list.append(transforms.RandomHorizontalFlip())
        train_transform_list.append(transforms.RandomCrop(im_size))
        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))
        train_data_transform = transforms.Compose(train_transform_list)


        test_transform_list = [transforms.Resize((resize))]
        test_transform_list.append(transforms.CenterCrop(im_size))
        test_transform_list.append(transforms.ToTensor())
        test_transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))
        test_data_transform = transforms.Compose(test_transform_list)
    
    if args.evaluate:
        train_dataset = Recipe1M(args, test_data_transform, 'train')
    else:
        train_dataset = Recipe1M(args, train_data_transform, 'train')

    val_dataset = Recipe1M(args, test_data_transform, 'val')
    test_dataset = Recipe1M(args, test_data_transform, 'test')

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    print("len(test_dataset):", len(test_dataset))
    return train_dataset, val_dataset, test_dataset


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long()
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_fn(data):
    """ collate to consume and batchify recipe data
    """

    # Sort a data list by caption length (descending order).
    image, ing_labels, ids, titles, ingrs, instrs = zip(*data)

    if image[0] is not None:
        # Merge images (from tuple of 3D tensor to 4D tensor).
        image = torch.stack(image, 0)
    else:
        image = None
    title_targets = pad_input(titles)
    ingredient_targets = pad_input(ingrs)
    instruction_targets = pad_input(instrs)
    ing_labels = torch.stack(ing_labels, 0)

    return image, ing_labels, ids, title_targets, ingredient_targets, instruction_targets
