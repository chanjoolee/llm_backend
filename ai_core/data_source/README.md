# Prerequisites
1. PdfFileDataSource를 사용하기 위해 필요한 패키지를 설치합니다.
2. local test를 하기 위해서 opensearch docker container를 실행합니다.

## poppler 설치

### OSX
```bash
brew install poppler
```

### RockyOS
```bash
sudo dnf install poppler
```

### Windows  
1. [poppler github](https://github.com/oschwartz10612/poppler-windows)에서 Release를 다운로드 받습니다.
2. Program Files 디렉토리에 압축을 풉니다.
3. 환경변수 PATH에 Program Files/poppler/bin을 추가합니다.

### 설치확인
```bash 
pdftoppm -h
```


## tesseract 설치

### OSX
```bash
brew install tesseract
```

### RockyOS
```bash
sudo dnf install tesseract
```

### Windows
1. [tesseract github](https://github.com/UB-Mannheim/tesseract/wiki) 에서 exe파일을 다운로드 받아 실행합니다.
2. 설치 과정중 additional language data에서 Korean을 선택합니다.

## Opensearch docker container 실행
ai_core/data_source/vectorstore/resources 위치로 이동합니다.

### 초기 비밀번호 설정
`.env` 파일을 열고, `OPENSEARCH_INITIAL_ADMIN_PASSWORD`를 설정합니다.

### docker-compose 실행
```bash
docker-compose -f docker-compose-dev.yml up -d