import importlib
from abc import ABC, abstractmethod
from typing import List, Dict

from camel.typing import ModelType
from data_analysis.chat_env import ChatEnv
from data_analysis.phase import Phase
from data_analysis.utils import log_visualize, check_bool


class ComposedPhase(ABC):
    def __init__(self,
                 phase_name: str,
                 cycle_num: int,
                 composition: List[Dict],
                 config_phase: Dict,
                 config_role: Dict,
                 model_type: ModelType,
                 log_filepath: str):
        """
        Initialize a composed phase with its configuration.

        Args:
            phase_name: Name of this composed phase
            cycle_num: Number of cycles to execute
            composition: List of phase configurations in this composed phase
            config_phase: Phase configurations
            config_role: Role configurations
            model_type: Model type to use
            log_filepath: Path to log file
        """
        self.phase_name = phase_name
        self.cycle_num = cycle_num
        self.composition = composition
        self.config_phase = config_phase
        self.config_role = config_role
        self.model_type = model_type
        self.log_filepath = log_filepath

        # Initialize role prompts
        self.role_prompts = {}
        for role in self.config_role:
            self.role_prompts[role] = "\n".join(self.config_role[role])

        # Initialize SimplePhase instances
        self.phases = []
        for phase_item in self.composition:
            phase_class = getattr(importlib.import_module("data_analysis.phase"), phase_item["phase"])
            phase_instance = phase_class(
                assistant_role_name=self.config_phase[phase_item["phase"]]["assistant_role_name"],
                user_role_name=self.config_phase[phase_item["phase"]]["user_role_name"],
                phase_prompt="\n".join(self.config_phase[phase_item["phase"]]["phase_prompt"]),
                role_prompts=self.role_prompts,
                phase_name=phase_item["phase"],
                model_type=self.model_type,
                log_filepath=self.log_filepath
            )
            self.phases.append(phase_instance)

    @abstractmethod
    def update_phase_env(self, chat_env: ChatEnv):
        """
        Update the phase environment from the global chat environment.

        Args:
            chat_env: Global chat environment
        """
        pass

    @abstractmethod
    def update_chat_env(self, chat_env: ChatEnv) -> ChatEnv:
        """
        Update the global chat environment with results from this phase.

        Args:
            chat_env: Global chat environment

        Returns:
            Updated chat environment
        """
        pass

    @abstractmethod
    def break_cycle(self, phase_env: Dict) -> bool:
        """
        Determine if the cycle should be broken early.

        Args:
            phase_env: Current phase environment

        Returns:
            True if cycle should break, False otherwise
        """
        pass

    def execute(self, chat_env: ChatEnv) -> ChatEnv:
        """
        Execute the composed phase.

        Args:
            chat_env: Global chat environment

        Returns:
            Updated chat environment
        """
        self.update_phase_env(chat_env)
        
        for cycle_index in range(self.cycle_num):
            for phase, phase_item in zip(self.phases, self.composition):
                max_turn_step = phase_item.get('max_turn_step', -1)
                need_reflect = check_bool(phase_item.get('need_reflect', 'False'))
                
                log_visualize(
                    f"**[Execute Detail]**\n\n"
                    f"Executing SimplePhase: {phase.phase_name} "
                    f"in ComposedPhase: {self.phase_name}, "
                    f"Cycle: {cycle_index + 1}/{self.cycle_num}"
                )
                
                chat_env = phase.execute(chat_env, max_turn_step, need_reflect)
                if self.break_cycle(phase.phase_env):
                    break
                    
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class DataPreparation(ComposedPhase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env: ChatEnv):
        self.phase_env.update({
            "data_understanding": chat_env.env_dict.get("data_understanding", "")
        })

    def update_chat_env(self, chat_env: ChatEnv) -> ChatEnv:
        chat_env.env_dict["data_preparation"] = self.phases[-1].seminar_conclusion
        return chat_env

    def break_cycle(self, phase_env: Dict) -> bool:
        return False


class Analysis(ComposedPhase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env: ChatEnv):
        self.phase_env.update({
            "data_preparation": chat_env.env_dict.get("data_preparation", "")
        })

    def update_chat_env(self, chat_env: ChatEnv) -> ChatEnv:
        chat_env.env_dict["analysis"] = self.phases[-1].seminar_conclusion
        return chat_env

    def break_cycle(self, phase_env: Dict) -> bool:
        return False


class ModelDevelopment(ComposedPhase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env: ChatEnv):
        self.phase_env.update({
            "analysis": chat_env.env_dict.get("analysis", "")
        })

    def update_chat_env(self, chat_env: ChatEnv) -> ChatEnv:
        chat_env.env_dict["model_development"] = self.phases[-1].seminar_conclusion
        return chat_env

    def break_cycle(self, phase_env: Dict) -> bool:
        return False


class Testing(ComposedPhase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env: ChatEnv):
        self.phase_env.update({
            "model_development": chat_env.env_dict.get("model_development", "")
        })

    def update_chat_env(self, chat_env: ChatEnv) -> ChatEnv:
        chat_env.env_dict["testing"] = self.phases[-1].seminar_conclusion
        return chat_env

    def break_cycle(self, phase_env: Dict) -> bool:
        return False