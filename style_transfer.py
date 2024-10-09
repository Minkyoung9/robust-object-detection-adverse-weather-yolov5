import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# 1. 이미지 전처리: 이미지를 텐서로 변환
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert("RGB")
    
    # 이미지를 max_size로 resize
    size = max_size if max(image.size) > max_size else max(image.size)
    
    if shape:
        size = shape
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225)) # VGG의 normalization 값
                             ])
    
    image = transform(image)[:3,:,:].unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)
    return image

# 2. 이미지 후처리 : 텐서를 이미지로 변환
def tensor_to_image(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)  # 배치 차원 제거
    image = transforms.ToPILImage()(image)
    return image

# 3. VGG19 모델의 특정 레이어에서 특징을 추출하는 함수
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',  # 첫 번째 합성곱 레이어의 결과
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content loss를 위해 사용할 레이어
            '28': 'conv5_1'
        }
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

# 4. 스타일 이미지의 그람 행렬 계산 함수
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())  # d x d 크기의 그람 행렬
    return gram / (d* h * w) # 출력값 정규화

# 5. 스타일 전이 수행 ; 학습 반복 step = 에폭 수 
def style_transfer(content_img, style_img, model, content_weight=1e4, style_weight=1e2, steps=300):
    # 콘텐츠와 스타일 이미지의 특징 추출
    content_features = get_features(content_img, model)
    style_features = get_features(style_img, model)
    
    # 스타일 이미지의 그람 행렬 계산
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # 생성할 이미지를 콘텐츠 이미지와 동일한 크기로 초기화
    target = content_img.clone().requires_grad_(True).to(device)
    
    #최적화 설정
    optimizer = optim.Adam([target], lr=0.003)
    
    for i in range(steps):
        target_features = get_features(target, model)
        
        # 콘텐츠 손실: 콘텐츠 이미지와 생성 이미지의 특정 레이어 차이
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        # 스타일 손실: 그람 행렬을 사용하여 스타일 이미지와 생성 이미지의 레이어 간 차이
        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature) # 정규화된 그람 행렬
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # 최적화 단계
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 50 스텝마다 중간 결과 출력
        if i % 50 == 0:
            print(f"Step {i}, Total loss: {total_loss.item()}")
    
    return target

# Start ------------------------------------------------------------------

# 6. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 7. 사전 학습된 VGG19 모델 로드 (특징 추출을 위한 모델)
vgg = models.vgg19(pretrained=True).features
vgg.to(device)

# 모델의 학습 방지 (freeze)
for param in vgg.parameters():
    param.requires_grad = False

# 8. 콘텐츠 및 스타일 이미지 로드
content_img = load_image('/home/intern/minkyoung/dataset/DAWN/train').to(device)
style_img = load_image('path_to_your_style_image.jpg', shape=content_img.shape[-2:]).to(device)

# 9. 스타일 전이 수행
output = style_transfer(content_img, style_img, vgg)

# 10. 결과 이미지 시각화
final_image = tensor_to_image(output)
plt.imshow(final_image)
plt.axis('off')
plt.show()

# 결과 이미지 저장
final_image.save("stylized_output.png")
