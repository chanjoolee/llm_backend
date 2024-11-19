import requests
import json
from langchain_core.tools import tool

@tool
def get_dashboard_panel_json(dashboard_uid: str, panel_id: int) -> str:
    """
    Grafana 대시보드의 특정 패널 JSON을 문자열로 반환
    오류 발생시 오류 메시지를 반환
    
    Args:
        dashboard_uid: 대시보드 UID
        panel_id: 패널 ID
    
    Returns:
        패널 JSON 문자열 또는 오류 메시지
    """
    GRAFANA_URL = "http://swm-02-01:3000"  # 여기를 실제 URL로 변경
    API_TOKEN = "your-api-token"  # 여기를 실제 토큰으로 변경

    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }

    try:
        url = f"{GRAFANA_URL}/api/dashboards/uid/{dashboard_uid}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        dashboard = response.json()

        for panel in dashboard.get('dashboard', {}).get('panels', []):
            if panel.get('id') == panel_id:
                return json.dumps(panel, indent=2, ensure_ascii=False)

        return f"패널을 찾을 수 없습니다. (dashboard_uid: {dashboard_uid}, panel_id: {panel_id})"

    except requests.exceptions.RequestException as e:
        return f"API 요청 오류: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON 파싱 오류: {str(e)}"
    except Exception as e:
        return f"예기치 않은 오류: {str(e)}"
