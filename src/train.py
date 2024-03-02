
class ProjectAgent:   
    def __init__(self):
        self.model = []
        self.device = 'cpu'
    
    def act(self, observation, use_random=False):
        return self.greedy_action(observation)

    def save(self, path):
        pass

    def load(self):
        self.model = torch.load(os.path.join(os.getcwd(),'Model_v1_PatientRef_E500.pth'), map_location=self.device)

    def greedy_action(self, observation):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to('cpu'))
            return torch.argmax(Q).item()



