# A Multi-dimensional Aggregation Network Guided by Key Features for Plaque Echo Classification Based on Carotid Ultrasound Video

## 📁 Project Structure

```
|- video_X_frame_120/
|  |- Mixed/
|  |  |- 2/
|  |  |  |- 0.jpg, 1.jpg, ..., N.jpg
|  |- Stable/
|  |- Unstable/
```

---

## 🧾 Dataset Format

Example (video with label):

```
Mixed/277/0.jpg ... Mixed/277/119.jpg | 2
Mixed/131/0.jpg ... Mixed/131/119.jpg | 2
```

### File Paths

Defined in `main.py`:

```python
train_k = 'annotation/avg_K5/{}_train.txt'.format(i)
test_k = 'annotation/avg_K5/{}_val.txt'.format(i)
```

Set the frame path accordingly:

```bash
--frame_path /YOUR_ROOT_DATASET/video_X_frame_120/
```

---

## ⚙️ Dataset Configuration (CNN-LSTM)

Structure:

```
|- Dataset/
|  |- plaque/
|  |  |- jpg/
|  |  |  |- Mixed/
|  |  |  |  |- 2/
|  |  |  |  |  |- 000000.jpg, ...
|  |- KFold/
|  |- labels/
```

Set paths in `dataset_config.py`:

```python
ROOT_DATASET = '/YOUR_ROOT_DATASET/'
root_data = ROOT_DATASET + 'plaque/jpg'
filename_imglist_train = 'plaque/KFold/{}_train.txt'.format(k)
filename_imglist_val = 'plaque/KFold/{}_val.txt'.format(k)
```

---

## 🏋️‍♂️ Model Training Scripts

### MA-Net / MFA-Net / MFH-Net

- Train using ResNet stages:

```bash
sh scripts/new_train_res4.sh 0
```

- ResNet backbone options:

```bash
sh scripts/new_train_res2.sh 0
sh scripts/new_train_res3.sh 0
sh scripts/new_train_res4.sh 0
sh scripts/new_train_res5.sh 0
sh scripts/new_train_res2_3.sh 0
sh scripts/new_train_res2_3_4.sh 0
sh scripts/new_train_res3_4_5.sh 0
```

- K-Fold + Cross-Slice Strategies:

```bash
sh scripts/new_train_KF2_CS_4_1.sh 0
sh scripts/new_train_KF3_CS_5.sh 0
sh scripts/new_train_CS_KF.sh 0
```

### CNN-LSTM / Temporal Models

```bash
python main.py --use_cuda
sh train_tsn_plaque.sh         # TSN
sh train_tsm_plaque.sh         # TSM
python train.py --modelName=C3D
python train.py --modelName=R3D
sh train.sh                    # AIA
sh train_tam_plaque.sh         # TAM
sh train_tan_plaque.sh         # TEA
sh train_tcm.sh 0              # TCM
sh train_tdn.sh                # TDN
sh train_ctnet.sh              # CT-Net
```

---

## Gram Matrix Visualization

Run:

```bash
python gram.py
```
