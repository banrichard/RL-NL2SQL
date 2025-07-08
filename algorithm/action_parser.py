from typing import Tuple, Any, Dict

def parse_action(
        action: Tuple[str, Any],
        current_components: Dict
) -> Dict:
    """
    解析一个结构化的动作，并更新SQL组件状态。

    参数:
    - action (Tuple[str, Any]): 一个元组，第一个元素是命令，第二个是参数。
      例如: ("SELECT_COLUMN", "name"), ("FROM_TABLE", "users")
    - current_components (Dict): 当前已构建的SQL组件字典。

    返回:
    - Dict: 更新后的SQL组件字典。
    """
    command, value = action
    updated_components = current_components.copy()

    if command == "SELECT_COLUMN":
        if value not in updated_components["select"]:
            updated_components["select"].append(value)
    elif command == "FROM_TABLE":
        # 简单起见，假设只有一个FROM表
        updated_components["from"] = value
    elif command == "WHERE_CONDITION":
        # value 应该是一个 (column, operator, value) 的元组
        updated_components["where"].append(value)
    elif command == "FINISH":
        # FINISH 动作用于表示SQL构建完成
        pass
    else:
        # 可以选择抛出错误或忽略未知命令
        print(f"警告: 未知的结构化命令 '{command}'")

    return updated_components


def build_sql_from_components(components: Dict) -> str:
    """辅助函数：从组件字典构建最终的SQL字符串"""
    if not components["select"] or not components["from"]:
        return "Error: Incomplete SQL"

    select_str = f"SELECT {', '.join(components['select'])}"
    from_str = f"FROM {components['from']}"
    where_str = ""
    if components["where"]:
        conditions = []
        for col, op, val in components["where"]:
            # 对值进行简单的类型判断，如果是字符串则加上引号
            if isinstance(val, str):
                val = f"'{val}'"
            conditions.append(f"{col} {op} {val}")
        where_str = "WHERE " + " AND ".join(conditions)

    return " ".join(filter(None, [select_str, from_str, where_str]))


if __name__ == "__main__":
    # --- 使用示例 ---
    # 假设RL策略依次生成了以下动作
    actions = [
        ("SELECT_COLUMN", "name"),
        ("SELECT_COLUMN", "age"),
        ("FROM_TABLE", "users"),
        ("WHERE_CONDITION", ("age", ">", 25)),
        ("FINISH", None)
    ]

    # 初始化SQL组件
    sql_components = {"select": [], "from": None, "where": []}

    for act in actions:
        if act[0] == "FINISH":
            break
        sql_components = parse_action(act, sql_components)
        # 打印每一步之后的状态
        # print("Updated Components:", sql_components)

    # 构建最终SQL
    final_sql = build_sql_from_components(sql_components)
    print(f"最终构建的SQL: {final_sql}")  # 输出: SELECT name, age FROM users WHERE age > 25
