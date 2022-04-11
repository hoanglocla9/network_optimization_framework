import json


def encodeColor(colorList, fullColorList):
    result = ["0"] * len(fullColorList)
    for color in colorList:
        colorIdx = fullColorList.index(color)
        result[colorIdx] = "1"
    return "".join(result)
            
def loadJSON(jsonFile):
    with open(jsonFile) as config:
        data = config.read()
    config = json.loads(data)
    return config
         
