# vehicle_tracking
[notion](https://hungvtm.notion.site/Vehicles-Tracking-ec0e338d5019450096fc3706697af953?pvs=4)

# Installation

- Install Virtual Environment
```bash
python -m venv venv
```

- Activate environment
```bash
venv\Scripts\activate
```

- Install requirement libraries
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

- To check torch gpu is available

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

- Run test get youtube stream

```bash
python app/test_get_yt_stream.py
```

- Run webapp

```
streamlit run app/webapp.py
```

# Reference

[Thử làm Object Tracking với YOLO v9 và DeepSORT](https://www.youtube.com/watch?v=hzOU9lp4Xng&t=802s&ab_channel=M%C3%ACAI)