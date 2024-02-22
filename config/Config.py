import json
import os


class Config:

    filePath = "config.json"

    def __init__(self):
        self.config = {
            "showFrameBorders": True,
            "MLPs": []
        }
        if os.path.exists(self.filePath):
            self.load()
        else:
            self.save()

    def load(self):
        with open(self.filePath, 'r') as file:
            self.config = json.load(file)

    def save(self):
        with open(self.filePath, 'w') as file:
            json.dump(self.config, file, indent=2)

    def isShowFrameBorders(self):
        return self.config["showFrameBorders"]

    def setShowFrameBorders(self, value: bool):
        self.config["showFrameBorders"] = value
        self.save()

    def getMLPStructures(self):
        return self.config["MLPs"]

    def getMLPStructureNames(self):
        return [d["Name"] for d in self.config["MLPs"]]

    def addMLPStructure(self, structure: dict):
        self.config["MLPs"].append(structure)
        self.save()


instance = Config()
