class BaseAgent:

    def __init__(self, env, simulation_app):
        self.env = env
        self.simulation_app = simulation_app
    
    def run_loop(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_action(self, obs, infos):
        raise NotImplementedError("This method should be overridden by subclasses.")
    