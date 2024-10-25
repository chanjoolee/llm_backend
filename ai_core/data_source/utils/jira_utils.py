import json
from typing import Iterable

from atlassian import Jira

from ai_core.data_source.utils.utils import safe_get, clean_json_string


def filter_required_fields(issue_fields: dict):
    required_fields = [
        "summary",
        "fixVersions",
        "resolution",
        "priority",
        "labels",
        "versions",
        "status",
        "components",
        "subtasks",
        "progress",
        "worklog",
        "issuetype",
        "project",
        "resolutiondate",
        "created",
        "updated",
        "description",
        "comment",
        "assignee"
    ]

    return {field: issue_fields[field] for field in required_fields if field in issue_fields}


def cleanse_fields(fields: dict):
    result = {}

    result["summary"] = safe_get(fields, "summary", "")

    assignee = safe_get(fields, "assignee", {})
    result["assignee"] = {"displayName": safe_get(assignee, "displayName", ""), "name": safe_get(assignee, "name", "")}

    result["fixVersions"] = [{"name": safe_get(fix_version, "name"), "released": safe_get(fix_version, "released")} for fix_version in fields.get("fixVersions", [])]

    resolution = safe_get(fields, "resolution", {})
    result["resolution"] = {"name": safe_get(resolution, "name"), "description": safe_get(resolution, "description")}

    priority = safe_get(fields, "priority", {})
    result["priority"] = safe_get(priority, "name")

    result["labels"] = safe_get(fields, "labels", [])
    result["versions"] = [{"name": safe_get(version, "name"), "released": safe_get(version, "released"), "releaseDate": safe_get(version, "releaseDate")} for version in fields.get("versions", [])]

    status = safe_get(fields, "status", {})
    status_category = safe_get(status, "statusCategory", {})
    result["status"] = {"description": safe_get(status, "description"), "name": safe_get(status, "name"), "statusCategory": safe_get(status_category, "name")}

    result["components"] = [{"name": safe_get(component, "name"), "description": safe_get(component, "description")} for component in fields.get("components", [])]
    result["subtasks"] = fields.get("subtasks", [])
    result["progress"] = fields.get("progress", {})
    result["worklog"] = fields.get("worklog", {})

    issue_type = safe_get(fields, "issuetype", {})
    result["issueType"] = {"name": safe_get(issue_type, "name"), "description": safe_get(issue_type, "description")}

    project = safe_get(fields, "project", {})
    project_category = safe_get(project, "projectCategory", {})
    result["project"] = {"name": safe_get(project, "name"), "description": clean_json_string(safe_get(project_category, "description"))}

    result["resolutionDate"] = fields.get("resolutiondate", None)
    result["created"] = fields.get("created", None)
    result["updated"] = fields.get("updated", None)
    result["description"] = clean_json_string(fields.get("description", None))

    comment = safe_get(fields, "comment", {})
    result["comments"] = [{"body": clean_json_string(safe_get(c, "body")), "displayName": safe_get(safe_get(c, "updateAuthor", {}), "displayName"), "created": safe_get(c, "created"), "updated": safe_get(c, "updated")} for c in safe_get(comment, "comments", [])]

    return result


def search_issues(url: str, project_key: str, access_token: str, assignee: str, start_date: str,
                  end_date: str) -> Iterable[str]:
    jira = Jira(url=url, token=access_token)
    issues = jira.jql(f"project = {project_key} AND assignee = {assignee} AND updated >= {start_date} AND updated <= {end_date}")

    for issue in issues.get("issues", []):
        required_fields = filter_required_fields(issue["fields"])
        cleaned_fields = cleanse_fields(required_fields)
        cleaned_fields["key"] = issue["key"]
        key = {"key": issue["key"]}

        yield json.dumps({**key, **cleaned_fields}, indent=2, ensure_ascii=False)