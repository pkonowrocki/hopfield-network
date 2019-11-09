import csv
import random

def corruptAllFigures(allFigures, percent, resultSizePerFigure):
    corruptedAllFigures = []
    for figure in allFigures:
        figure = figure.copy()
        corruptedFigures = corruptFigure(figure, percent, resultSizePerFigure)
        corruptedAllFigures.append(corruptedFigures)
    return corruptedAllFigures

def corruptFigure(figure, percent, resultSize):
    # duplicate the same figure resultSize times to modify each in a different way
    figures = []
    for i in range(resultSize):
        figures.append(figure.copy())

    return _corruptData(figures, percent)

def corruptData():
    filename = 'large-25x25'
    inititalFigures = _readData(filename)
    corruptedFigures = _corruptData(inititalFigures, _getPercent())
    _saveData(corruptedFigures, filename)
    return corruptedFigures;

def _readData(filename):
    fullPath = f'data/{filename}.csv' 
    figures = []
    with open(fullPath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for figure in csv_reader:
            figures.append(figure)
    print(f'Read data from \'{fullPath}\'.')
    return figures

def _getPercent():
    percent = int(input('What is expected amount of mutations as percent of total amount of pixels? (0 - 100)\n'))
    if percent < 0 or percent > 100:
        print('Invalid value!')
        return
    return percent

def _corruptData(figures, percent):
    itemsToCorrupt = round(percent / 100 * len(figures[0]))
    # print(f'itemsToCorrupt: {itemsToCorrupt}')
    for i, figure in enumerate(figures):
        # print(f'Modifying figure #{i+1}')
        alreadyMutated = []
        mutationIndex = None
        for j in range(itemsToCorrupt):
            while mutationIndex is None or mutationIndex in alreadyMutated:
                mutationIndex = random.randrange(0, len(figure))
            figure[mutationIndex] = -1 * int(figure[mutationIndex])
            alreadyMutated.append(mutationIndex)
            # print(f' {j+1}/{itemsToCorrupt} mutated on index: {mutationIndex}')
    return figures

def _saveData(figures, originalFilename):
    newFilename = originalFilename + ''
    fullPath = f'data/{newFilename}-corrupted.csv' 
    with open(fullPath, 'w+', newline='\n', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for figure in figures:
            csv_writer.writerow(figure)
    print(f'Saved data to \'{fullPath}\'.')

if __name__ == "__main__":
    corruptData()
