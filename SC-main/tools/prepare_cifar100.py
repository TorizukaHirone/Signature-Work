import os, random
from torchvision.datasets import CIFAR100
from PIL import Image

random.seed(0)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out_root = os.path.join(ROOT, "data_use", "cifar-100")
train_dir = os.path.join(out_root, "train_pro")
test_dir = os.path.join(out_root, "test_pro")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_set = CIFAR100(root=os.path.join(ROOT, "data_download"), train=True, download=True)
test_set  = CIFAR100(root=os.path.join(ROOT, "data_download"), train=False, download=True)

classes = train_set.classes

for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)

def dump(dataset, split_dir, list_path, sample_ratio=1.0):
    idxs = list(range(len(dataset)))
    if sample_ratio < 1.0:
        k = int(len(idxs) * sample_ratio)
        idxs = random.sample(idxs, k)

    lines = []
    for i in idxs:
        img, label = dataset[i]
        cname = classes[label]
        fname = f"{i:06d}.png"
        rel_path = os.path.join(cname, fname)  # relative to data_prefix
        abs_path = os.path.join(split_dir, rel_path)
        # img may be PIL.Image or numpy array depending on torchvision version
        if hasattr(img, "save"):      # PIL.Image
            img.save(abs_path)
        else:                         # numpy array
            Image.fromarray(img).save(abs_path)

        lines.append(f"{rel_path} {label}\n")

    with open(list_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

dump(train_set, train_dir, os.path.join(out_root, "train_10p.list"), sample_ratio=0.1)
dump(test_set,  test_dir,  os.path.join(out_root, "test.list"), sample_ratio=1.0)

print("Done:", out_root)