import os
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
#  1. 模拟的数据库 (用于演示)
# ----------------------------------------------------------------------
# 在真实应用中，这些工具会去查询 PeeringDB, CAIDA, 或 IXPDB
# ----------------------------------------------------------------------

# 模拟 PeeringDB (所有权)
MOCK_PEERING_DB = {
    15169: "Google LLC",
    32934: "Google LLC",  # 15169 和 32934 是 Sibling
    701: "Verizon Business",
    1239: "Sprint",
    2906: "Netflix",
    3257: "T-Mobile",
}

# 模拟 CAIDA AS-Relationships 数据库
# 使用 frozenset 确保 (A, B) 和 (B, A) 都能查到
MOCK_CAIDA_DB = {
    # 701 (Provider) -> 1239 (Customer)
    frozenset({701, 1239}): "provider-customer", 
    # 701 和 7018 (AT&T) 是 Peer
    frozenset({701, 7018}): "peer",
    # Netflix 和 T-Mobile 没有记录 (我们将用 IXP 来推断)
}

# 模拟 IXP 数据库
MOCK_IXP_DB = {
    "DE-CIX Frankfurt": {
        "type": "Public IXP",
        "members": [2906, 3257, 701]
    },
    "Equinix Ashburn": {
        "type": "Data Center (Private Peering)",
        "members": []
    },
    "Google Cloud": {
        "type": "Cloud Facility",
        "members": [15169, 32934]
    }
}

# ----------------------------------------------------------------------
#  2. 定义 Agent 可用的工具 (Tools)
# ----------------------------------------------------------------------

@tool
def get_as_owner(asn: int) -> str:
    """查询指定 ASN 的注册组织（所有者）名称。"""
    return MOCK_PEERING_DB.get(asn, "Unknown")

@tool
def get_caida_relationship(asn_a: int, asn_b: int) -> str:
    """查询 CAIDA 数据库中两个 AS 之间的关系 (peer, provider-customer)。"""
    key = frozenset({asn_a, asn_b})
    return MOCK_CAIDA_DB.get(key, "unknown")

@tool
def get_location_info(location_name: str) -> Dict[str, Any]:
    """查询指定地理位置的信息，例如它是否是一个公共 IXP 及其成员列表。"""
    return MOCK_IXP_DB.get(location_name, {"type": "Unknown", "members": []})

# ----------------------------------------------------------------------
#  3. Agent 定义 (LLM, Prompt, 和 Agent 本身)
# ----------------------------------------------------------------------

def create_as_relationship_agent():
    """创建并返回 BGP 关系分析 Agent"""
    
    # 使用 gpt-4o 或 gpt-4-turbo。Agent 需要强大的模型
    # temperature=0 确保 Agent 严格遵循逻辑，而不是“发挥创意”
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # 绑定工具
    tools = [get_as_owner, get_caida_relationship, get_location_info]

    # Agent 的核心逻辑 (System Prompt)
    # 这就是你设计的“提示词工作”，它告诉 LLM 如何 *使用* 工具
    system_prompt = """
    你是一个 BGP 网络拓扑专家。
    你的任务是根据用户输入 (AS A, AS B, loc) 确定 AS A 和 AS B 之间的关系。
    关系类型必须是: `sibling`, `peer`, `provider-customer`, 或 `unknown`。

    你 **必须** 严格遵循以下决策流程（优先级由高到低）：

    1.  **[P1: 检查 Sibling 关系]**
        * 分别调用 `get_as_owner` 查询 AS A 和 AS B 的所有者。
        * 如果两个所有者相同，且不是 "Unknown"，则关系 **必定** 是 `sibling`。立即返回 `sibling` 并停止后续步骤。

    2.  **[P2: 检查权威数据库]**
        * 如果不是 `sibling`，调用 `get_caida_relationship` 查询它们的关系。
        * 如果返回 `peer` 或 `provider-customer`，则使用该结果。立即返回该结果并停止。

    3.  **[P3: 检查地点 (IXP 启发式)]**
        * 如果 CAIDA 数据库返回 `unknown`，则检查 `loc`。
        * 调用 `get_location_info(loc)`。
        * 如果 `type` 是 "Public IXP"，并且 AS A 和 AS B **都** 在 `members` 列表（或即使列表为空，地点本身也强烈暗示），则关系 *很可能* 是 `peer`。
        * 在这种情况下，返回 `peer` (并可注明是基于 IXP 推断的)。

    4.  **[P4: LLM 知识库]**
        * 如果以上所有步骤都返回 `unknown`（例如，CAIDA 未知，地点也不是公共 IXP），此时才允许你使用你自己的（训练数据中的）储备知识进行“最佳猜测”。
        * 例如，即时工具信息不足，你也可能知道两个著名的 Tier-1 运营商在 "Equinix Ashburn" 这样的私有设施中进行 `peer`。
        * 你的最终回答 **必须** 包含你得出结论的**完整推理链**。

    请开始工作。
    """

    # 创建 Prompt 模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "请分析: (AS A: {asn_a}, AS B: {asn_b}, Location: '{loc}')"),
            ("placeholder", "{agent_scratchpad}"), # 存储 Agent 的思考过程
        ]
    )
    
    # 创建 Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 创建 Agent 执行器 (包含执行循环)
    # verbose=True 会打印出 Agent 的 "思考->行动->观察" 循环
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return executor

# ----------------------------------------------------------------------
#  4. 封装和运行
# ----------------------------------------------------------------------

class AnalysisInput(BaseModel):
    asn_a: int
    asn_b: int
    loc: str
        
def analyze_relationship(agent_executor: AgentExecutor, asn_a: int, asn_b: int, loc: str) -> str:
    """
    运行 Agent 分析并返回最终结果。
    """
    print(f"\n--- 正在分析: (AS A: {asn_a}, AS B: {asn_b}, Location: '{loc}') ---\n")
    
    # 准备 Agent 的输入
    input_data = {
        "asn_a": asn_a,
        "asn_b": asn_b,
        "loc": loc
    }
    
    # executor.invoke 会运行完整的 "思考-行动-观察" 循环
    response = agent_executor.invoke(input_data)
    
    print(f"\n--- 分析结束 ---\n")
    return response['output']


# --- 主程序入口 ---
if __name__ == "__main__":
    # 确保你已设置 OPENAI_API_KEY 环境变量
    if "OPENAI_API_KEY" not in os.environ:
        print("错误: 请先设置 OPENAI_API_KEY 环境变量。")
    else:
        # 1. 创建 Agent
        my_agent = create_as_relationship_agent()
        
        # 2. 运行不同场景的分析

        # 场景 1: Sibling 关系 (P1 命中)
        # 15169 (Google) 和 32934 (Google)
        analyze_relationship(my_agent, 15169, 32934, "Google Cloud")
        
        # 场景 2: Provider-Customer (P2 命中)
        # 701 (Verizon) 和 1239 (Sprint), 在 MOCK_CAIDA_DB 中有记录
        analyze_relationship(my_agent, 701, 1239, "New York")

        # 场景 3: Peer (IXP 启发式, P3 命中)
        # 2906 (Netflix) 和 3257 (T-Mobile) 在 CAIDA 中 *没有* 记录
        # 但它们都在 DE-CIX Frankfurt (一个 IXP)
        analyze_relationship(my_agent, 2906, 3257, "DE-CIX Frankfurt")

        # 场景 4: Peer (LLM 储备知识, P4 命中)
        # 701 (Verizon) 和 7018 (AT&T)。假设 CAIDA 数据库 *缺失* 这条记录
        # (我们的 mock db 中设置了 peer，但我们假设它缺失了来测试 P4)
        # Agent 在 P1, P2, P3 都失败后，应依靠 "Tier-1 互为 Peer" 的储备知识
        analyze_relationship(my_agent, 701, 7018, "Equinix Ashburn")
