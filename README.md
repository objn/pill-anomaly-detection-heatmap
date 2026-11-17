# Deep Learning-Based Unsupervised Pill Defect Detection and Localization

โครงงานทดลองใช้ Convolutional Autoencoder (CAE) ตรวจจับและระบุตำแหน่งความผิดปกติบนภาพเม็ดยาจากชุดข้อมูล MVTec AD หมวด `pill` โดยฝึกจากภาพปกติ แล้วใช้ความคลาดเคลื่อนระหว่างภาพจริงกับภาพที่โมเดลสร้างกลับเป็น anomaly score และ heatmap

> โปรเจกต์นี้เป็น baseline สำหรับการศึกษาและเปรียบเทียบ loss ยังไม่ใช่ระบบตรวจสอบคุณภาพที่ผ่านการรับรองหรือพร้อมใช้ในสายการผลิต

## แนวทาง

- **การเรียนรู้:** Unsupervised anomaly detection; ใช้ภาพปกติ (`train/good`) ฝึก CAE
- **การประเมิน:** ใช้ภาพ test และ ground-truth masks เปรียบเทียบผลระดับภาพและระดับพิกเซล
- **Loss ที่เปรียบเทียบ:** MSE, SSIM และ Gradient Difference Loss (GDL)
- **การเตรียมภาพ:** resize เป็น 256 × 256 พิกเซล และปรับค่า pixel เป็นช่วง 0–1
- **การแบ่งข้อมูล:** กันภาพปกติ 15% จาก train เป็น validation; ใช้ validation เลือก threshold ที่ percentile 95 ก่อนประเมิน test
- **Threshold:** คะแนนภาพที่เกินหรือเท่ากับ threshold ถูกจัดเป็นภาพผิดปกติ

## ไฟล์ทดลอง

| Notebook | Loss |
|---|---|
| `main-MSE.ipynb` | Mean Squared Error (MSE) |
| `main-SSIM.ipynb` | Structural Similarity (SSIM) |
| `main-GDL.ipynb` | Gradient Difference Loss (GDL) |

แต่ละ notebook มีขั้นตอนเตรียมข้อมูล ฝึก CAE สร้าง anomaly heatmap และคำนวณ metric โปรดรันเซลล์ตามลำดับ โดยเซลล์แบ่ง train/validation ต้องทำก่อนฝึกโมเดล

## ผลการทดลองเบื้องต้น

ตัวเลขต่อไปนี้ถอดจากรายงาน baseline เป็นผลการทดลองเดิม ไม่ใช่ผลที่คำนวณใหม่จากทุกเครื่องหรือทุกการรัน

| วิธี | Specificity | Recall | Image AUROC | Pixel AUROC |
|---|---:|---:|---:|---:|
| CAE (MSE) | 0.961538 | 0.184397 | 0.711948 | 0.819182 |
| CAE (SSIM) | 0.923077 | 0.198582 | 0.703764 | 0.863058 |
| CAE (GDL) | 0.961538 | 0.021277 | 0.346836 | 0.189468 |

ในผลเบื้องต้น SSIM ให้ Pixel AUROC สูงสุด ส่วน MSE ให้ Image AUROC สูงสุดในสามแบบที่เปรียบเทียบ GDL ให้ผลต่ำ โดยเฉพาะ Recall และ AUROC ทั้งสองระดับ รายงานเสนอว่าการใช้ขนาด gradient โดยไม่รักษาทิศทางของ gradient อาจทำให้ภาพสร้างกลับคลาดเคลื่อน และทำให้คะแนนความผิดปกติบนพื้นหลังรบกวนผลได้

Metric ที่รายงานในตารางข้างต้นมี Specificity, Recall, Image AUROC และ Pixel AUROC; notebook ยังอาจแสดง Precision และ F1 จาก threshold ที่เลือกไว้ด้วย ผลเหล่านั้นขึ้นกับการตั้ง threshold และข้อมูลที่ใช้ประเมิน

## การประเมินผล

- **Image AUROC:** วัดความสามารถในการจัดอันดับภาพปกติและภาพผิดปกติจากคะแนนภาพ โดยรวมหลาย threshold
- **Precision, Recall, F1 และ Specificity:** คำนวณหลังใช้ image threshold ที่เลือกจาก validation
- **Pixel AUROC:** วัดการจัดอันดับคะแนน anomaly ของแต่ละพิกเซลเทียบกับ ground-truth mask โดยไม่ต้องใช้ threshold

ชุดข้อมูล MVTec AD ไม่มี validation split แยกมาให้ โครงการจึงกันภาพปกติจากชุด train มาทำ validation ตามรายงาน เพื่อกำหนด threshold จากการกระจายคะแนนภาพปกติ วิธีนี้ไม่มีภาพเสียใน validation จึงไม่ได้เลือก threshold ที่ทำให้ F1 สูงสุดจาก validation

## เริ่มใช้งาน

### 1. เตรียมข้อมูล

ดาวน์โหลด MVTec Anomaly Detection Dataset และวางโฟลเดอร์หมวด `pill` ไว้ข้างไฟล์ notebook:

```text
project/
├── main-MSE.ipynb
├── main-SSIM.ipynb
├── main-GDL.ipynb
└── pill/
    ├── train/good/
    ├── test/
    └── ground_truth/
```

ชุดข้อมูลไม่ได้อยู่ใน Git repository นี้ ตรวจสอบเงื่อนไขการใช้งานและการอ้างอิงของ MVTec AD ก่อนนำข้อมูลหรือผลไปใช้

### 2. เตรียม Python environment

โปรเจกต์ทดลองบน Python 3.11 และ PyTorch ที่รองรับ CUDA ตัวอย่างคำสั่งใช้ `uv` ใน PowerShell:

```powershell
uv python install 3.11
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install torch torchvision --torch-backend cu128
uv pip install jupyter ipykernel matplotlib pandas scikit-learn pillow
```

เปิด notebook ด้วย kernel จาก `.venv` แล้วตรวจ GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not found")
```

โน้ตบุ๊กกำหนด `SEED = 42` และใช้ GPU ตัวแรกที่ CUDA มองเห็น หากต้องการให้ผลเทียบกันได้ ให้คง seed, data split, จำนวน epoch และ hyperparameters เหมือนกันในทุก loss

## Hyperparameters จากรายงาน

- Learning rate: `2e-4`
- Weight decay: `1e-5`
- Epochs: `100`
- Seed: `42`
- Validation fraction: `15%` ของภาพปกติจาก train
- Image size: `256 × 256`

## ข้อจำกัดและงานต่อ

- ผลเป็น baseline จากชุดข้อมูลเฉพาะหมวด pill; ไม่ได้ยืนยันการทำงานกับยา ภาพ หรือสภาพแสงอื่น
- Reconstruction error อาจเด่นที่พื้นหลังหรือขอบเม็ดยา ทำให้ตำหนิขนาดเล็กถูกกลบ
- ผล GDL เบื้องต้นต่ำกว่าวิธีอื่นและควรปรับวิธีคำนวณ loss/การจัดการพื้นหลังก่อนสรุปประสิทธิภาพ
- รายงานเสนอให้ทดลองภาพขนาด 512 × 512 และการตัดพื้นหลังก่อนเข้าโมเดล
- การสาธิตผ่านหน้าเว็บ/กล้องเรียลไทม์และการติดตั้งในสายการผลิตยังไม่ใช่ขอบเขตการประเมินใน repository นี้

## อ้างอิง

- Bergmann, P. et al. (2019). [A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html). CVPR.
- Bergmann, P. et al. (2018). [Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders](https://arxiv.org/abs/1807.02011).
- Wang, Z. et al. (2004). [Image Quality Assessment: From Error Visibility to Structural Similarity](https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html).
- Mathieu, M., Couprie, C., and LeCun, Y. (2016). [Deep multi-scale video prediction beyond mean square error](https://arxiv.org/abs/1511.05440).
