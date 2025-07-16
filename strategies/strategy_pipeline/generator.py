import os
import configparser
import random
from strategies.strategy_pipeline.utils.indicator_utils import get_available_indicators, get_enabled_indicators, random_select_indicators, IndicatorConfig

class StrategyConfigLoader:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_general(self):
        return self.config['general']

    def get_data(self):
        return self.config['DATA']

    def get_limits(self):
        return self.config['limits']

    def get_indicator_sections(self):
        # Return all sections that are not general, DATA, or limits
        return [s for s in self.config.sections() if s not in ['general', 'DATA', 'limits']]

    def get_section(self, section):
        return dict(self.config[section])

class StrategyBuilder:
    def __init__(self, base_filename, prefix, output_dir):
        self.base_filename = base_filename
        self.prefix = prefix
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_and_save(self, strategy_dict, idx):
        filename = f"{self.prefix}{self.base_filename}_{idx}.ini"
        filepath = os.path.join(self.output_dir, filename)
        config = configparser.ConfigParser()
        for section, values in strategy_dict.items():
            config[section] = values
        with open(filepath, 'w') as f:
            config.write(f)
        return filepath

class StrategyGenerator:
    def __init__(self, config_path, output_dir):
        self.loader = StrategyConfigLoader(config_path)
        self.output_dir = output_dir
        self.indicator_catalog = get_available_indicators()

    def generate(self):
        general = self.loader.get_general()
        data = self.loader.get_data()
        limits = self.loader.get_limits()
        max_files = int(limits.get('max_strategy_files', 10))
        base_filename = general.get('base_filename', 'strategy')
        prefix = general.get('prefix', 'strat_')
        builder = StrategyBuilder(base_filename, prefix, self.output_dir)

        # For demo: randomly select 3-5 indicators per strategy
        for idx in range(1, max_files + 1):
            strategy = {
                'general': dict(general),
                'DATA': dict(data),
                'indicators': {}
            }
            # For now, enable all available indicators
            enabled = list(self.indicator_catalog.keys())
            n_ind = random.randint(3, 5)
            selected = random_select_indicators(enabled, n_ind)
            for ind_name in selected:
                ind_cfg = self.indicator_catalog[ind_name]
                window = ind_cfg.random_window()
                if window:
                    strategy['indicators'][ind_name] = str(window)
                else:
                    strategy['indicators'][ind_name] = 'enabled'
            builder.build_and_save(strategy, idx)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'strategy_config.ini')
    output_dir = os.path.dirname(__file__)
    generator = StrategyGenerator(config_path, output_dir)
    generator.generate()
