import os, sys, pdb
from shutil import copyfile
from xml.etree import ElementTree

dict_of_moves = ['Serve Forehand Backspin',
                'Serve Forehand Loop',
                'Serve Forehand Sidespin',
                'Serve Forehand Topspin',

                'Serve Backhand Backspin',
                'Serve Backhand Loop',
                'Serve Backhand Sidespin',
                'Serve Backhand Topspin',

                'Offensive Forehand Hit',
                'Offensive Forehand Loop',
                'Offensive Forehand Flip',

                'Offensive Backhand Hit',
                'Offensive Backhand Loop',
                'Offensive Backhand Flip',

                'Defensive Forehand Push',
                'Defensive Forehand Block',
                'Defensive Forehand Backspin',

                'Defensive Backhand Push',
                'Defensive Backhand Block',
                'Defensive Backhand Backspin']

if __name__ == "__main__":
    try:
        name_folder = str(sys.argv[1])
    except:
        raise ValueError('\n\nUsage : python verif_xml_files.py folder_with_xmls')

    for file in [file for file in os.scandir('test') if file.name[-3:] == 'xml']:
        print(file.name)

        tree = ElementTree.parse(file.path)
        root = tree.getroot()
        test_actions = []
        for action in root:
            test_actions.append([int(action.get('begin')), int(action.get('end'))])
        test_actions.sort()

        tree = ElementTree.parse(os.path.join(name_folder, file.name))
        root = tree.getroot()
        output_actions = []
        for action in root:
            output_actions.append([int(action.get('begin')), int(action.get('end')), action.get('move')])
        output_actions.sort()

        if len(test_actions) != len(output_actions):
            raise ValueError('The xmls do not have the same number of actions')

        for test_action, output_action in zip(test_actions, output_actions):
            if test_action[0] != output_action[0]:
                raise ValueError('The begin frame of the actions has been modified %s' % file.name)

            if test_action[1] != output_action[1]:
                raise ValueError('The end frame of the actions has been modified in %s' % file.name)

            if output_action[2] not in dict_of_moves:
                raise ValueError('The move associated to the action in %s is not in the possible moves : %s not in : ' % (file.name, output_action[2]), dict_of_moves)

    print('xml files have the correct format. Thank you for your participation.')
