# daisy_backend



## Getting started

### 파이썬 패키지 설치
pip install -r requirements.txt

pip install -e ./subtree/langgraph-checkpoint-mysql

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.tde.sktelecom.com/ORYX/daisy/daisy_backend.git
git branch -M main
git push -uf origin main
```


## Langchain 업그레이드

### Langchain v0.3.3, langgraph v0.2.35

#### 작업 순서

- 기존 대화 테이블 삭제 
  - daisy.checkpoints와 daisy.writes 테이블 드랍 
  - 백엔드 대화 테이블 레코드 삭제
- 파이썬 패키지 업그레이드
  - requirements.txt 설치
- langgraph-checkpoint-mysql subtree 설치
  - pip install -e ./subtree/langgraph-checkpoint-mysql
- 비동기 호출 방식으로 백엔드 코드 수정 
  - 대화 제목 자동 생성
  - 대화 삭제
  - 대화 복제
- langgraph 신규 대화 테이블 생성 
  - ai_core.checkpoint.setup.py 참고
  - checkpoint_migrations, checkpoints, checkpoint_blobs, checkpoint_writes 테이블 생성 확인 
