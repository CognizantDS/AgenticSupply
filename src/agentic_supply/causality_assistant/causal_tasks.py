from agentic_supply.causality_assistant.causal_analysis import CausalModel


class CausalTask:
    def __init__(self, causal_model: CausalModel):
        self.causal_model = causal_model

    def identify_causal_effect(self):
        identified_estimand = self.causal_model.model.identify_effect()
