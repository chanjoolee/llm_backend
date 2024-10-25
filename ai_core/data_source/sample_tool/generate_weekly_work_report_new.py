from langchain_core.tools import tool

from ai_core.data_source.utils.jira_utils import search_issues


@tool
def get_weekly_tde_jira_issues_tool(assignee: str, start_date: str, end_date: str):
    '''
    This is a tool for get weekly tde jira issues.
    param project_names str: Comma separated Jira project names,  e.g. "ORYX,DATAENG,SWIFT"
    param assignee str: Jira assignee name, e.g. "1113593"
    param start_date str: Start date of updated for searching issues, e.g. "2024-09-01"
    param end_date str: End date of updated for searching issues, e.g. "2024-09-30"
    '''

    def get_project_issues(url: str, access_token: str, assignee:str, start_date: str, end_date: str,
                           projects: list[str]) -> str:
        project_issues = ""
        for project in projects:
            issues = search_issues(url, project, access_token, assignee, start_date, end_date)

            project_issues += f"==============JIRA PROJECT: {project}==============\n{list(issues)}\n\n"

        return project_issues

    url = "https://jira.tde.sktelecom.com"
    access_token = "NDY2NDU2NDQ5Nzg5OvPiaqUWQeggMNIjtBWj669jY87N"
    projects = ["ORYX", "DATAENG", "SWIFT"]
    tde_project_issues = get_project_issues(url, access_token, assignee, start_date, end_date, projects)

    url = "https://doss.sktelecom.com/jira"
    access_token = "MTU4Mzg4NTAwMTIxOu9CJafvMgKEfLJo69G4u65OAljv"
    projects = ["SMZ000375"] # SMZ000375: DATAPORTAL_PROD
    doss_project_issues = get_project_issues(url, access_token, assignee, start_date, end_date, projects)

    return tde_project_issues + doss_project_issues
