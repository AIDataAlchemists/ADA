import importlib
import os
import re
from abc import ABC, abstractmethod

from camel.agents import RolePlaying
from camel.messages import ChatMessage
from camel.typing import TaskType, ModelType
from data_analysis.chat_env import ChatEnv
from data_analysis.utils import log_visualize


class Phase(ABC):
    def __init__(self,
                 assistant_role_name,
                 user_role_name,
                 phase_prompt,
                 role_prompts,
                 phase_name,
                 model_type,
                 log_filepath):
        """
        Args:
            assistant_role_name: who receives chat in a phase
            user_role_name: who starts the chat in a phase
            phase_prompt: prompt of this phase
            role_prompts: prompts of all roles
            phase_name: name of this phase
        """
        self.seminar_conclusion = None
        self.assistant_role_name = assistant_role_name
        self.user_role_name = user_role_name
        self.phase_prompt = phase_prompt
        self.phase_env = dict()
        self.phase_name = phase_name
        self.assistant_role_prompt = role_prompts[assistant_role_name]
        self.user_role_prompt = role_prompts[user_role_name]
        self.model_type = model_type
        self.log_filepath = log_filepath

    @abstractmethod
    def update_phase_env(self, chat_env):
        """
        Update the phase environment from the global chat environment
        """
        pass

    @abstractmethod
    def update_chat_env(self, chat_env) -> ChatEnv:
        """
        Update the global chat environment with the results from this phase
        """
        pass

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        """
        Execute the chatting in this phase
        """
        self.update_phase_env(chat_env)
        role_play_session = RolePlaying(
            assistant_role_name=self.assistant_role_name,
            user_role_name=self.user_role_name,
            assistant_role_prompt=self.assistant_role_prompt,
            user_role_prompt=self.user_role_prompt,
            task_prompt=chat_env.env_dict['task_prompt'],
            task_type=TaskType.CHATDEV,
            with_task_specify=False,
            model_type=self.model_type,
            background_prompt=chat_env.config.background_prompt
        )

        _, input_user_msg = role_play_session.init_chat(None, self.phase_env, self.phase_prompt)
        assistant_response, user_response = role_play_session.step(input_user_msg, chat_turn_limit == 1)

        conversation_meta = "**{}<->{} on : {}**\n\n".format(self.assistant_role_name, self.user_role_name, self.phase_name)
        log_visualize(role_play_session.assistant_agent.role_name, conversation_meta + role_play_session.assistant_sys_msg.content)
        log_visualize(role_play_session.user_agent.role_name, conversation_meta + role_play_session.user_sys_msg.content)

        for _ in range(chat_turn_limit):
            if isinstance(assistant_response.msg, ChatMessage):
                log_visualize(role_play_session.assistant_agent.role_name, conversation_meta + assistant_response.msg.content)
                if role_play_session.assistant_agent.info:
                    self.seminar_conclusion = assistant_response.msg.content
                    break
                if assistant_response.terminated:
                    break

            if isinstance(user_response.msg, ChatMessage):
                log_visualize(role_play_session.user_agent.role_name, conversation_meta + user_response.msg.content)
                if role_play_session.user_agent.info:
                    self.seminar_conclusion = user_response.msg.content
                    break
                if user_response.terminated:
                    break

        chat_env = self.update_chat_env(chat_env)
        return chat_env


class DomainUnderstanding(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        pass  # No specific environment updates needed for this phase

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['domain_understanding'] = self.seminar_conclusion
        return chat_env


class DataUnderstanding(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['domain_understanding'] = chat_env.env_dict.get('domain_understanding', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['data_understanding'] = self.seminar_conclusion
        return chat_env


class DataCleaning(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['data_understanding'] = chat_env.env_dict.get('data_understanding', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['data_cleaning'] = self.seminar_conclusion
        return chat_env


class DataPipeline(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['data_cleaning'] = chat_env.env_dict.get('data_cleaning', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['data_pipeline'] = self.seminar_conclusion
        return chat_env


class ExploratoryAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['data_pipeline'] = chat_env.env_dict.get('data_pipeline', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['exploratory_analysis'] = self.seminar_conclusion
        return chat_env


class StatisticalAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['exploratory_analysis'] = chat_env.env_dict.get('exploratory_analysis', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['statistical_analysis'] = self.seminar_conclusion
        return chat_env


class FeatureEngineering(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['statistical_analysis'] = chat_env.env_dict.get('statistical_analysis', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['feature_engineering'] = self.seminar_conclusion
        return chat_env


class ModelSelection(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['feature_engineering'] = chat_env.env_dict.get('feature_engineering', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['model_selection'] = self.seminar_conclusion
        return chat_env


class ModelTraining(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['model_selection'] = chat_env.env_dict.get('model_selection', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['model_training'] = self.seminar_conclusion
        return chat_env


class ModelValidation(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['model_training'] = chat_env.env_dict.get('model_training', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['model_validation'] = self.seminar_conclusion
        return chat_env


class HyperparameterTuning(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['model_validation'] = chat_env.env_dict.get('model_validation', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['hyperparameter_tuning'] = self.seminar_conclusion
        return chat_env


class ModelInterpretability(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['hyperparameter_tuning'] = chat_env.env_dict.get('hyperparameter_tuning', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['model_interpretability'] = self.seminar_conclusion
        return chat_env


class CodeReview(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['model_interpretability'] = chat_env.env_dict.get('model_interpretability', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['code_review'] = self.seminar_conclusion
        return chat_env


class TestExecution(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['code_review'] = chat_env.env_dict.get('code_review', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['test_execution'] = self.seminar_conclusion
        return chat_env


class ErrorSummary(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['test_execution'] = chat_env.env_dict.get('test_execution', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['error_summary'] = self.seminar_conclusion
        return chat_env


class CorrectionImplementation(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['error_summary'] = chat_env.env_dict.get('error_summary', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['correction_implementation'] = self.seminar_conclusion
        return chat_env


class Reporting(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env['correction_implementation'] = chat_env.env_dict.get('correction_implementation', '')

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['report'] = self.seminar_conclusion
        return chat_env