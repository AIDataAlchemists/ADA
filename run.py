import argparse
import logging
import os
import sys

from data_analysis.chat_chain import ChatChain
from camel.typing import ModelType

def main():
    parser = argparse.ArgumentParser(description='Data Analysis System')
    parser.add_argument('--config', type=str, default="Default",
                        help="Name of config, which is used to load configuration under CompanyConfig/")
    parser.add_argument('--org', type=str, default="DefaultOrganization",
                        help="Name of organization, your software will be generated in WareHouse/name_org_timestamp")
    parser.add_argument('--task', type=str, default="Analyze the sales data to identify trends and patterns",
                        help="Prompt of software")
    parser.add_argument('--name', type=str, default="SalesAnalysis",
                        help="Name of software, your software will be generated in WareHouse/name_org_timestamp")
    parser.add_argument('--model', type=str, default="GPT_3_5_TURBO",
                        help="GPT Model, choose from {'GPT_3_5_TURBO', 'GPT_4', 'GPT_4_TURBO', 'GPT_4O', 'GPT_4O_MINI'}")
    parser.add_argument('--path', type=str, default="",
                        help="Your file directory, ChatDev will build upon your software in the Incremental mode")
    parser.add_argument('--database', type=str, default="sales_data.db",
                        help="Database or dataset selection")
    
    args = parser.parse_args()

    args2type = {'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
                 'GPT_4': ModelType.GPT_4,
                 'GPT_4_TURBO': ModelType.GPT_4_TURBO,
                 'GPT_4O': ModelType.GPT_4O,
                 'GPT_4O_MINI': ModelType.GPT_4O_MINI,
                 }
    
    if openai_new_api:
        args2type['GPT_3_5_TURBO'] = ModelType.GPT_3_5_TURBO_NEW

    chat_chain = ChatChain(config_path=os.path.join("CompanyConfig", args.config, "ChatChainConfig.json"),
                           config_phase_path=os.path.join("CompanyConfig", args.config, "PhaseConfig.json"),
                           config_role_path=os.path.join("CompanyConfig", args.config, "RoleConfig.json"),
                           task_prompt=args.task,
                           project_name=args.name,
                           org_name=args.org,
                           model=args2type[args.model],
                           code_path=args.path)

    logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                        format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

    chat_chain.pre_processing()
    chat_chain.make_recruitment()
    chat_chain.execute_chain()
    chat_chain.post_processing()

if __name__ == "__main__":
    main()