from atlassian import Confluence
from langchain_core.tools import tool


@tool
def upload_weekly_worklog_to_confluence_comment_tool(page_title: str, content: str):
    '''
    This is a tool for uploading data to Confluence.
    Content will be uploaded to the page's comment section.

    param page_title: Confluence page title
    param content: Content to be uploaded to page's comment section
    '''

    url = "https://confluence.tde.sktelecom.com/"
    access_token = "NTQ3NzM0NjkxOTE5Oqj+0NWVhXAv7VRPxgLrE2ir8VnW"
    space = "DATAENG"


    confluence = Confluence(url=url, token=access_token)
    page_exists = confluence.page_exists(space, page_title, type=None)

    if not page_exists:
        return f"Page {page_title} does not exist in space {space}"
    else:
        page_id = confluence.get_page_id(space, page_title)
        confluence.add_comment(page_id, content)

        return f"Content uploaded to page {page_title} in space {space}"
