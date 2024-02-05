from streamlit_drawable_canvas import st_canvas
import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1*1*120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.fc1(self.flat(x)))
        x = self.fc2(x)
        return x

def import_and_predict(img, model):
  img_transform = transforms.Compose([transforms.Grayscale(), transforms.RandomInvert(p=1)])
  img_new = img_transform(img)
  composed = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
  img_t = composed(img_new)
  img_t = img_t.type(torch.float32)
  x = img_t.expand(1, 1, 28, 28)
  z = model(x)
  z = nn.Softmax(dim=1)(z)
  p_max, yhat = torch.max(z.data, 1)
  p = float(format(p_max.numpy()[0], '.4f'))*100
  yhat = int(float(yhat.numpy()[0]))
  st.success(f"작성하신 숫자는 {p:.2f} %로 {yhat}로 예측됩니다.")

def load_model():
  model = LeNet5()
  model.load_state_dict(torch.load('./LeNet5_20240129.pth'))
  return model

model = load_model()
st.write("""
         # MNIST를 사용한 숫자 인식 예측 앱
        """)
st.markdown("""
            ## 숫자를 그려보세요!
            """)
st.write("Note: 숫자가 캔버스의 대부분을 차지하고 캔버스 중앙에 위치하도록 이미지를 그립니다.")
st.sidebar.header("사용자 입력")

# Specify brush parameters and drawing mode
b_width = st.sidebar.slider("브러시 너비 선택: ", 1, 100, 10)
b_color = st.sidebar.color_picker("브러시 색상 16진수 입력: ")
bg_color = st.sidebar.color_picker("배경색 16진수 입력: ", "#FFFFFF")

# Create a canvas component
canvas = st_canvas(
    stroke_width=b_width,
    stroke_color=b_color,
    background_color=bg_color,
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas.image_data is not None:
    image = Image.fromarray(canvas.image_data)
    w, h = image.size
    st.image(image, width=500, caption="숫자 이미지")
    import_and_predict(image, model)