import time

from opensearchpy import AsyncOpenSearch, OpenSearch


def create_opensearch_index_name(prefix: str) -> str:
    return f"{prefix}_{str(int(time.time()))}"


def switch_to_new_index(opensearch_client: OpenSearch, index_alias: str, index_name: str, prev_index_names: list[str]) \
        -> None:
    """
    새로운 index로 alias를 변경하고, 기존 index를 삭제한다.

    :param opensearch_client: OpenSearch client
    :param index_alias: alias 이름
    :param index_name: 새로운 index 이름
    :param prev_index_names: 이전 index 이름들
    """

    if prev_index_names:
        for prev_index_name in prev_index_names:
            # update alias atomically
            opensearch_client.indices.update_aliases(
                body={"actions": [
                    {"remove": {"index": prev_index_name, "alias": index_alias}},
                    {"add": {"index": index_name, "alias": index_alias}}]})

            # 기존 index 삭제
            if prev_index_name != index_name and (opensearch_client.indices.exists(index=prev_index_name)):
                opensearch_client.indices.delete(index=prev_index_name)
    else:
        opensearch_client.indices.put_alias(index=index_name, name=index_alias)


async def aswitch_to_new_index(opensearch_client: AsyncOpenSearch, index_alias: str, index_name: str, prev_index_names: list[str]) \
        -> None:
    """
    새로운 index로 alias를 변경하고, 기존 index를 삭제한다.

    :param opensearch_client: OpenSearch client
    :param index_alias: alias 이름
    :param index_name: 새로운 index 이름
    :param prev_index_names: 이전 index 이름들
    """

    if prev_index_names:
        for prev_index_name in prev_index_names:
            # update alias atomically
            await opensearch_client.indices.update_aliases(
                body={"actions": [
                    {"remove": {"index": prev_index_name, "alias": index_alias}},
                    {"add": {"index": index_name, "alias": index_alias}}]})

            # 기존 index 삭제
            if prev_index_name != index_name and (await opensearch_client.indices.exists(index=prev_index_name)):
                await opensearch_client.indices.delete(index=prev_index_name)
    else:
        await opensearch_client.indices.put_alias(index=index_name, name=index_alias)