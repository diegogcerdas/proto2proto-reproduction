import yaml

class Arguments:
    
    class ProtoPNet:

        def __init__(self, args_dict_protoPNet):
            self.numPrototypes = args_dict_protoPNet['numPrototypes']
            self.W1 = args_dict_protoPNet['W1']
            self.H1 = args_dict_protoPNet['H1']

    class Backbone:

        def __init__(self, args_dict_backbone):
            self.name = args_dict_backbone['name']
            self.pretrained = args_dict_backbone['pretrained']
            self.loadPath = args_dict_backbone['loadPath']

    class Dataloader:

        def __init__(self, args_dict_dataloader):
            self.testDir = args_dict_dataloader['testDir']
            self.testBatchSize = args_dict_dataloader['testBatchSize']
            self.numWorkers = args_dict_dataloader['numWorkers']

    def __init__(self, filename):
        with open(filename, "r") as file:
            args_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.dataLoader = self.Dataloader(args_dict['dataLoader'])
        args_dict = args_dict['settingsConfig']
        self.imgSize = args_dict['imgSize']
        self.numFeatures = args_dict['numFeatures']
        self.PrototypeActivationFunction = args_dict['PrototypeActivationFunction']
        self.backbone = self.Backbone(args_dict['backbone'])
        self.protoPNet = self.ProtoPNet(args_dict['protoPNet'])