from gitlab import Gitlab
from gitlab.v4.objects import Project

# GitLab API 토큰과 프로젝트 ID 설정
GITLAB_TOKEN = "tde2-c4gLGAsrvixkqMg-KBoL"
PROJECT_ID = "SWIFT/streams"

# GitLab API 기본 URL
BASE_URL = "https://gitlab.tde.sktelecom.com"


def get_file_content(project: Project, file_path: str, sha: str):
    """특정 파일의 내용 가져오기"""
    # 파일 내용 가져오기 (base64로 인코딩되어 있음)
    try:
        file_obj = project.files.get(file_path=file_path, ref=sha)

        content = file_obj.decode()

        return content.decode('utf-8')
    except Exception as e:
        return None


def get_code_context(file_content: str, start_line: int, end_line: int):
    """코드 컨텍스트 출력"""
    lines = file_content.split('\n')
    context = lines[max(0, start_line - 1):min(len(lines), end_line)]

    texts = "\n"
    for i, line in enumerate(context, start=start_line):
        texts += f"{i}: {line}\n"

    return texts


def generate_discussion_texts():
    if not GITLAB_TOKEN or not PROJECT_ID:
        raise ValueError("GITLAB_TOKEN과 GITLAB_PROJECT_ID 환경 변수를 설정해주세요.")

    gl = Gitlab(url=BASE_URL, private_token=GITLAB_TOKEN)
    project = gl.projects.get(PROJECT_ID)
    merge_requests = list(project.mergerequests.list(all=True))
    # merge_header = f"총 {len(merge_requests)}개의 Merge Request를 찾았습니다."

    discussion_texts = []
    for mr in merge_requests:
        # merge_header = f"\nMerge Request #{mr.iid}: {mr.title}"
        discussions = list(mr.discussions.list(all=True))

        for discussion in discussions:
            note = discussion.attributes['notes'][0]
            if note['system']:
                continue

            author = note['author']['name']
            if author == "DevSecOps IT보안운영(임시)":
                continue

            header = f"Discussion ID: {discussion.id}"
            texts = ""
            if 'position' in note:
                position = note['position']
                file_path = position['new_path']
                start_line = position.get('new_line')
                if start_line is None:
                    continue

                start_line -= 5 if start_line > 5 else start_line
                end_line = start_line + 10  # 컨텍스트로 5줄 표시

                texts += f"\n    파일: {file_path}"
                file_content = get_file_content(project, file_path, mr.sha)
                if file_content:
                    texts += f"\n    코드 컨텍스트:"
                    texts += get_code_context(file_content, start_line, end_line)

            for note in discussion.attributes['notes']:
                texts += f"\n    작성자: {note['author']['name']}"
                texts += f"\n    내용: {note['body']}"

            if texts:
                discussion_texts.append(header + texts)

    return discussion_texts


if __name__ == "__main__":
    discussions = generate_discussion_texts()
    for diss in discussions:
        print(diss)
        print()
    # print(len(discussions))
