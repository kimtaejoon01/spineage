# SpineAge Desktop

Electron + FastAPI + PyTorch 기반 척추 영상 AI 분석 애플리케이션입니다.

현재 저장소에는 소스 코드만 포함되며, 대용량 AI 모델 가중치는 GitHub에 포함하지 않습니다.
새 컴퓨터에서 실행하려면 아래 순서대로 환경을 준비해야 합니다.

## 1. 사전 요구사항

필수:

- Git
- Node.js LTS + npm
- Python 3.10 또는 3.11 권장
- 최초 패키지/사전학습 모델 다운로드를 위한 인터넷 연결
- AI 모델 가중치 파일

선택:

- NVIDIA GPU + CUDA 환경

GPU가 없어도 앱은 CPU로 실행할 수 있지만, AI 추론 및 TotalSegmentator 전처리는 상당히 느릴 수 있습니다.

> 중요: Electron에서 Python 경로를 프로젝트 루트의 `venv/`로 찾도록 되어 있으므로 가상환경 이름은 반드시 `venv`로 만드는 것을 권장합니다.

## 2. 프로젝트 받기

```bash
git clone https://github.com/kimtaejoon01/spineage.git
cd spineage
```

이미 clone한 저장소라면:

```bash
git pull origin main
```

## 3. Python 가상환경 만들기

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
py -m venv venv
venv\Scripts\activate
```

정상적으로 활성화되면 터미널 앞에 `(venv)`가 표시됩니다.

## 4. Python 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

주요 의존성에는 다음이 포함됩니다.

- FastAPI / Uvicorn
- PyTorch / torchvision
- pydicom
- SimpleITK
- h5py
- OpenCLIP
- transformers
- TotalSegmentator
- ACSConv

NVIDIA GPU를 사용할 경우 해당 컴퓨터의 CUDA 환경과 맞는 PyTorch 설치가 필요할 수 있습니다.

## 5. Node / Electron 패키지 설치

`package-lock.json`이 있으므로 새 환경에서는 다음을 권장합니다.

```bash
npm ci
```

필요한 경우 다음도 사용할 수 있습니다.

```bash
npm install
```

`node_modules/`는 GitHub에 포함되지 않습니다.

## 6. AI 모델 가중치 배치

모델 가중치는 용량 때문에 GitHub에 포함하지 않습니다.

프로젝트에 다음 폴더를 준비합니다.

```text
backend/
└── models/
```

아래 3개 파일이 필요합니다.

```text
backend/models/
├── use_clinical_0.pth
├── use_clinical_1.pth
└── checkpoint_best.pth
```

용도:

- `use_clinical_0.pth`: OVF 모델, 임상정보 미사용
- `use_clinical_1.pth`: OVF 모델, 임상정보 사용
- `checkpoint_best.pth`: HDF5 Multitask 모델

모델 파일은 https://drive.google.com/drive/folders/1889AUBICAAKOymLpGFGfjiYGgqqmTova 경로에서 다운받을 수 있습니다. 

모델이 없으면 서버 시작 또는 예측 시 `checkpoint not found` 오류가 발생합니다.

## 7. 최초 실행 시 추가 모델 다운로드

새 컴퓨터에서는 처음 실행하거나 특정 기능을 처음 사용할 때 추가 pretrained model 다운로드가 발생할 수 있습니다.

예:

- BiomedCLIP
- torchvision ConvNeXt pretrained weights
- TotalSegmentator weights

따라서 최초 환경 구성 시에는 인터넷 연결을 권장합니다.

완전 오프라인 PC에서 사용할 경우 필요한 pretrained 모델 캐시까지 미리 준비해야 합니다.

## 8. 실행

가상환경 `venv`가 활성화된 상태에서:

```bash
npm start
```

Electron이 실행되면서 내부적으로 Python FastAPI 서버도 같이 시작합니다.

정상 실행 예:

```text
OVF models pre-loaded & warm-up done
Routers mounted & OVF warm-up done
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
GET /health 200 OK
```

## 9. 백엔드 상태 확인

```bash
curl http://127.0.0.1:8000/health
```

HTTP 200 응답이 오면 백엔드가 정상 실행 중입니다.

## 10. GPU 동작

현재 SpineAge 자체 PyTorch 모델은 다음 순서로 device를 선택합니다.

```text
CUDA 사용 가능 -> CUDA
CUDA 사용 불가 -> CPU
```

따라서:

- NVIDIA CUDA GPU: CUDA 환경이 올바르게 설치되어 있으면 GPU 사용 가능
- macOS Apple Silicon: 현재 SpineAge 자체 모델 코드는 MPS를 직접 선택하지 않으므로 CPU로 실행
- GPU 없음: CPU 실행

GPU 환경에서는 PyTorch/CUDA 버전 호환성을 별도로 확인해야 합니다.

## 11. macOS / Windows / Linux 간 환경 공유 주의

`venv/`는 운영체제와 CPU 아키텍처에 종속적입니다.

따라서 기존 컴퓨터의 `venv/` 폴더를 다른 컴퓨터에 그대로 복사하지 말고, 대상 컴퓨터에서 새로 생성하세요.

예:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

다음 환경은 각각 별도로 준비해야 합니다.

- macOS Apple Silicon (arm64)
- macOS Intel (x64)
- Windows x64
- Linux x64

## 12. DICOM 입력 관련

OVF DICOM 로더는 가능한 경우 환자 공간 위치 정보를 사용해 슬라이스를 정렬하며, 메타데이터가 부족하면 다른 DICOM 태그 또는 업로드 순서를 사용합니다.

현재 OVF 로더는 single-frame 2D DICOM slice 시리즈를 기준으로 동작합니다.

DICOM 로드 실패 시 앱에서 표시되는 실제 오류 메시지를 확인하세요.

## 13. HDF5 Multitask 입력 관련

현재 체크포인트는 `channels_num_2d=3` 입력 구조를 기준으로 사용합니다.

HDF5 파일에는 다음 dataset이 필요합니다.

```text
image
muscle_mask
vertebrae_mask
```

세 dataset의 volume shape은 동일해야 합니다.

## 14. Electron 배포본 생성

개발 환경에서 정상 동작을 확인한 후:

```bash
npm run dist
```

현재 Electron Builder 설정은 다음을 배포 리소스에 포함하도록 되어 있습니다.

```text
electron/
backend/
venv/
```

따라서 배포 전에 해당 OS/CPU 환경에서 다음 준비가 끝나 있어야 합니다.

1. 프로젝트 루트에 `venv/` 생성
2. `pip install -r backend/requirements.txt`
3. `backend/models/`에 모델 가중치 배치
4. `npm ci`
5. 개발 실행 테스트

### macOS 배포 주의

현재 `package.json`의 macOS 빌드 설정은 `entitlements.mac.plist`를 참조합니다.
배포용 macOS 빌드를 만들기 전에는 해당 entitlement 파일과 코드서명 설정을 점검해야 합니다.

또한 `venv`는 OS/CPU 아키텍처에 종속되므로 macOS에서 만든 Python 환경을 Windows 배포에 그대로 사용할 수 없습니다.

## 15. GitHub에 포함하지 않는 파일

다음 항목은 의도적으로 Git에서 제외합니다.

```text
node_modules/
venv/
.venv/
dist/
build/
out/
release/
backend/models/
backend/workspace/
.env
.cache/
```

따라서 다른 컴퓨터에서는 `node_modules`, `venv`, 모델 파일 등을 직접 준비해야 합니다.

## Troubleshooting

### `OVF model checkpoint not found`

다음을 확인하세요.

```text
backend/models/use_clinical_0.pth
backend/models/use_clinical_1.pth
```

### `HDF5 model checkpoint not found`

다음을 확인하세요.

```text
backend/models/checkpoint_best.pth
```

### Electron 실행 직후 Python backend가 종료됨

가상환경 위치를 확인하세요.

macOS / Linux:

```bash
ls venv/bin/python
```

Windows:

```powershell
Get-Item venv\Scripts\python.exe
```

그리고 패키지를 다시 설치합니다.

```bash
pip install -r backend/requirements.txt
```

### Electron은 뜨지만 API가 연결되지 않음

```bash
curl http://127.0.0.1:8000/health
```

200 응답이 오지 않으면 `npm start`를 실행한 터미널의 Python 오류 로그를 확인하세요.

### 새 컴퓨터에서 첫 실행이 오래 걸림

BiomedCLIP, ConvNeXt, TotalSegmentator 등 추가 pretrained weight 다운로드 또는 초기 모델 warm-up이 진행 중일 수 있습니다.

## 개발 실행 최소 체크리스트

```text
[ ] Git / Node.js / Python 설치
[ ] 프로젝트 clone
[ ] 프로젝트 루트에 venv 생성
[ ] Python requirements 설치
[ ] npm ci
[ ] backend/models/에 모델 3개 배치
[ ] 최초 실행 시 인터넷 연결
[ ] npm start
[ ] /health 200 확인
```
